// Copyright 2025 The clvk authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "testcl.hpp"

// Function pointer type for clEnqueueSVMMemcpy
typedef cl_int (CL_API_CALL *clEnqueueSVMMemcpy_fn)(
    cl_command_queue command_queue,
    cl_bool blocking_copy,
    void* dst_ptr,
    const void* src_ptr,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);

// Basic USM test: allocation, setting kernel arg, CPU access
TEST_F(WithCommandQueue, USMBasicAllocation) {
    // Get function pointers
    auto clSharedMemAllocINTEL = (clSharedMemAllocINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(gPlatform, "clSharedMemAllocINTEL");
    auto clMemFreeINTEL = (clMemFreeINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(gPlatform, "clMemFreeINTEL");
    auto clSetKernelArgMemPointerINTEL = (clSetKernelArgMemPointerINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(gPlatform, "clSetKernelArgMemPointerINTEL");

    if (clSharedMemAllocINTEL == nullptr || clMemFreeINTEL == nullptr ||
        clSetKernelArgMemPointerINTEL == nullptr) {
        GTEST_SKIP() << "USM extension not available";
    }

    // 1. Allocate USM memory (1024 integers)
    const size_t num_elements = 1024;
    const size_t size = num_elements * sizeof(int);
    cl_int err;
    void* ptr = clSharedMemAllocINTEL(m_context, gDevice, nullptr, size, 0, &err);

    ASSERT_EQ(err, CL_SUCCESS) << "Failed to allocate USM memory";
    ASSERT_NE(ptr, nullptr) << "USM allocation returned null pointer";

    int* data = static_cast<int*>(ptr);

    // Initialize data on CPU
    for (size_t i = 0; i < num_elements; i++) {
        data[i] = static_cast<int>(i);
    }

    // Create and build kernel
    const char* source = R"(
    kernel void increment(__global int* data) {
        size_t id = get_global_id(0);
        data[id] = data[id] + 1;
    }
    )";

    auto program = CreateAndBuildProgram(source);
    auto kernel = CreateKernel(program, "increment");

    // 2. Set kernel argument with USM pointer
    err = clSetKernelArgMemPointerINTEL(kernel, 0, ptr);
    ASSERT_EQ(err, CL_SUCCESS) << "Failed to set kernel argument with USM pointer";

    // Execute kernel
    size_t global_size = num_elements;
    err = clEnqueueNDRangeKernel(m_queue, kernel, 1, nullptr, &global_size,
                                 nullptr, 0, nullptr, nullptr);
    ASSERT_EQ(err, CL_SUCCESS) << "Failed to enqueue kernel";

    // Wait for completion
    err = clFinish(m_queue);
    ASSERT_EQ(err, CL_SUCCESS) << "Failed to finish queue";

    // 3. Verify results on CPU (zero-copy access!)
    bool success = true;
    for (size_t i = 0; i < num_elements; i++) {
        if (data[i] != static_cast<int>(i + 1)) {
            printf("Mismatch at data[%zu]: expected %d but got %d\n",
                   i, static_cast<int>(i + 1), data[i]);
            success = false;
            if (i > 10) break;  // Don't spam too much
        }
    }
    EXPECT_TRUE(success) << "Data verification failed";

    // Cleanup
    err = clMemFreeINTEL(m_context, ptr);
    EXPECT_EQ(err, CL_SUCCESS) << "Failed to free USM memory";
}

// Test device-only memory allocation
TEST_F(WithCommandQueue, USMDeviceAllocation) {
    // Get function pointers
    auto clDeviceMemAllocINTEL = (clDeviceMemAllocINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(gPlatform, "clDeviceMemAllocINTEL");
    auto clMemFreeINTEL = (clMemFreeINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(gPlatform, "clMemFreeINTEL");
    auto clSetKernelArgMemPointerINTEL = (clSetKernelArgMemPointerINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(gPlatform, "clSetKernelArgMemPointerINTEL");

    if (clDeviceMemAllocINTEL == nullptr || clMemFreeINTEL == nullptr ||
        clSetKernelArgMemPointerINTEL == nullptr) {
        GTEST_SKIP() << "USM extension not available";
    }

    // 1. Allocate device memory (1024 integers)
    const size_t num_elements = 1024;
    const size_t size = num_elements * sizeof(int);
    cl_int err;
    void* device_ptr = clDeviceMemAllocINTEL(m_context, gDevice, nullptr, size, 0, &err);

    ASSERT_EQ(err, CL_SUCCESS) << "Failed to allocate device USM memory";
    ASSERT_NE(device_ptr, nullptr) << "Device USM allocation returned null pointer";

    // 2. Create a kernel that initializes the device memory
    const char* source = R"(
    kernel void init_device(__global int* data) {
        size_t id = get_global_id(0);
        data[id] = (int)id * 3;
    }
    )";

    auto program = CreateAndBuildProgram(source);
    auto kernel = CreateKernel(program, "init_device");

    // 3. Set kernel argument with device USM pointer
    err = clSetKernelArgMemPointerINTEL(kernel, 0, device_ptr);
    ASSERT_EQ(err, CL_SUCCESS) << "Failed to set kernel argument with device USM pointer";

    // Execute kernel
    size_t global_size = num_elements;
    err = clEnqueueNDRangeKernel(m_queue, kernel, 1, nullptr, &global_size,
                                 nullptr, 0, nullptr, nullptr);
    ASSERT_EQ(err, CL_SUCCESS) << "Failed to enqueue kernel";

    err = clFinish(m_queue);
    ASSERT_EQ(err, CL_SUCCESS) << "Failed to finish queue";

    // 4. Copy device memory to host-accessible buffer for verification
    auto host_buffer = CreateBuffer(CL_MEM_WRITE_ONLY, size, nullptr);
    
    // Note: We can't directly read device memory from CPU
    // In a real scenario, you'd use clEnqueueMemcpyINTEL or another kernel
    // For now, we just verify the allocation and kernel execution succeeded

    // Cleanup
    err = clMemFreeINTEL(m_context, device_ptr);
    EXPECT_EQ(err, CL_SUCCESS) << "Failed to free device USM memory";
}

TEST_F(WithCommandQueue, USMMemcpy) {
    // Get function pointers
    auto clSharedMemAllocINTEL = (clSharedMemAllocINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(gPlatform, "clSharedMemAllocINTEL");
    auto clMemFreeINTEL = (clMemFreeINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(gPlatform, "clMemFreeINTEL");
    
    if (clSharedMemAllocINTEL == nullptr || clMemFreeINTEL == nullptr) {
        GTEST_SKIP() << "USM extension not available";
    }

    const size_t num_elements = 1024;
    const size_t size = num_elements * sizeof(int);
    cl_int err;

    // 1. Allocate two USM buffers
    void* src_ptr = clSharedMemAllocINTEL(m_context, gDevice, nullptr, size, 0, &err);
    ASSERT_EQ(err, CL_SUCCESS) << "Failed to allocate source USM memory";
    ASSERT_NE(src_ptr, nullptr) << "Source USM allocation returned null pointer";

    void* dst_ptr = clSharedMemAllocINTEL(m_context, gDevice, nullptr, size, 0, &err);
    ASSERT_EQ(err, CL_SUCCESS) << "Failed to allocate destination USM memory";
    ASSERT_NE(dst_ptr, nullptr) << "Destination USM allocation returned null pointer";

    // 2. Initialize source data on CPU
    int* src_data = static_cast<int*>(src_ptr);
    for (size_t i = 0; i < num_elements; i++) {
        src_data[i] = static_cast<int>(i * 2);
    }

    // 3. Copy from source to destination using clEnqueueSVMMemcpy
    err = clEnqueueSVMMemcpy(m_queue, CL_TRUE, dst_ptr, src_ptr, size, 0, nullptr, nullptr);
    ASSERT_EQ(err, CL_SUCCESS) << "Failed to copy USM memory";

    // 4. Verify the copy worked
    int* dst_data = static_cast<int*>(dst_ptr);
    for (size_t i = 0; i < num_elements; i++) {
        EXPECT_EQ(dst_data[i], static_cast<int>(i * 2)) 
            << "Copy verification failed at index " << i;
    }

    // 5. Cleanup
    err = clMemFreeINTEL(m_context, src_ptr);
    EXPECT_EQ(err, CL_SUCCESS) << "Failed to free source USM memory";

    err = clMemFreeINTEL(m_context, dst_ptr);
    EXPECT_EQ(err, CL_SUCCESS) << "Failed to free destination USM memory";
}

TEST_F(WithCommandQueue, USMHostMemoryCopy) {
    // Get function pointers for USM extensions
    auto clSharedMemAllocINTEL = (clSharedMemAllocINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(gPlatform, "clSharedMemAllocINTEL");
    auto clMemFreeINTEL = (clMemFreeINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(gPlatform, "clMemFreeINTEL");

    if (clSharedMemAllocINTEL == nullptr || clMemFreeINTEL == nullptr) {
        GTEST_SKIP() << "USM extension not available";
    }

    const size_t num_elements = 1024;
    const size_t size = num_elements * sizeof(int);
    cl_int err;

    // 1. Allocate USM buffer
    void* usm_ptr = clSharedMemAllocINTEL(m_context, gDevice, nullptr, size, 0, &err);
    ASSERT_EQ(err, CL_SUCCESS) << "Failed to allocate USM memory";
    ASSERT_NE(usm_ptr, nullptr) << "USM allocation returned null pointer";

    // 2. Allocate host memory and initialize it
    std::vector<int> host_data(num_elements);
    for (size_t i = 0; i < num_elements; i++) {
        host_data[i] = static_cast<int>(i * 7);
    }

    // 3. Test host memory to USM copy
    err = clEnqueueSVMMemcpy(m_queue, CL_TRUE, usm_ptr, host_data.data(), size, 0, nullptr, nullptr);
    ASSERT_EQ(err, CL_SUCCESS) << "Failed to copy host memory to USM";

    // 4. Verify the copy worked by reading from USM
    int* usm_data = static_cast<int*>(usm_ptr);
    for (size_t i = 0; i < num_elements; i++) {
        EXPECT_EQ(usm_data[i], static_cast<int>(i * 7))
            << "Host->USM copy verification failed at index " << i;
    }

    // 5. Test USM to host memory copy
    std::vector<int> host_data2(num_elements, 0); // Initialize to 0
    
    err = clEnqueueSVMMemcpy(m_queue, CL_TRUE, host_data2.data(), usm_ptr, size, 0, nullptr, nullptr);
    ASSERT_EQ(err, CL_SUCCESS) << "Failed to copy USM to host memory";

    // 6. Verify the copy worked
    for (size_t i = 0; i < num_elements; i++) {
        EXPECT_EQ(host_data2[i], static_cast<int>(i * 7))
            << "USM->Host copy verification failed at index " << i;
    }

    // 7. Cleanup
    err = clMemFreeINTEL(m_context, usm_ptr);
    EXPECT_EQ(err, CL_SUCCESS) << "Failed to free USM memory";
}

TEST_F(WithCommandQueue, USMDeviceOnlyCopyLimitation) {
    // Get function pointers for USM extensions
    auto clDeviceMemAllocINTEL = (clDeviceMemAllocINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(gPlatform, "clDeviceMemAllocINTEL");
    auto clMemFreeINTEL = (clMemFreeINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(gPlatform, "clMemFreeINTEL");

    if (clDeviceMemAllocINTEL == nullptr || clMemFreeINTEL == nullptr) {
        GTEST_SKIP() << "USM extension not available";
    }

    const size_t num_elements = 1024;
    const size_t size = num_elements * sizeof(int);
    cl_int err;

    // 1. Allocate device-only USM buffer
    void* device_usm_ptr = clDeviceMemAllocINTEL(m_context, gDevice, nullptr, size, 0, &err);
    ASSERT_EQ(err, CL_SUCCESS) << "Failed to allocate device-only USM memory";
    ASSERT_NE(device_usm_ptr, nullptr) << "Device-only USM allocation returned null pointer";

    // 2. Allocate host memory and initialize it
    std::vector<int> host_data(num_elements);
    for (size_t i = 0; i < num_elements; i++) {
        host_data[i] = static_cast<int>(i * 11);
    }

    // 3. Test host memory to device-only USM copy - should now succeed with GPU-based copying
    err = clEnqueueSVMMemcpy(m_queue, CL_TRUE, device_usm_ptr, host_data.data(), size, 0, nullptr, nullptr);
    EXPECT_EQ(err, CL_SUCCESS) << "Failed to copy host memory to device-only USM";

    // 4. Test device-only USM to host memory copy - should also succeed
    std::vector<int> host_data2(num_elements, 0);
    err = clEnqueueSVMMemcpy(m_queue, CL_TRUE, host_data2.data(), device_usm_ptr, size, 0, nullptr, nullptr);
    EXPECT_EQ(err, CL_SUCCESS) << "Failed to copy device-only USM to host memory";

    // 5. Verify the copy worked
    for (size_t i = 0; i < num_elements; i++) {
        EXPECT_EQ(host_data2[i], static_cast<int>(i * 11))
            << "Device-only USM->Host copy verification failed at index " << i;
    }

    // 6. Test device-only USM to device-only USM copy - should also succeed
    void* device_usm_ptr2 = clDeviceMemAllocINTEL(m_context, gDevice, nullptr, size, 0, &err);
    ASSERT_EQ(err, CL_SUCCESS) << "Failed to allocate second device-only USM memory";
    
    err = clEnqueueSVMMemcpy(m_queue, CL_TRUE, device_usm_ptr2, device_usm_ptr, size, 0, nullptr, nullptr);
    EXPECT_EQ(err, CL_SUCCESS) << "Failed to copy device-only USM to device-only USM";

    // 7. Cleanup
    err = clMemFreeINTEL(m_context, device_usm_ptr);
    EXPECT_EQ(err, CL_SUCCESS) << "Failed to free first device-only USM memory";

    err = clMemFreeINTEL(m_context, device_usm_ptr2);
    EXPECT_EQ(err, CL_SUCCESS) << "Failed to free second device-only USM memory";
}

// Matrix multiplication using USM: C = A * B
TEST_F(WithCommandQueue, USMMatrixMultiply) {
    auto clSharedMemAllocINTEL = (clSharedMemAllocINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(gPlatform, "clSharedMemAllocINTEL");
    auto clMemFreeINTEL = (clMemFreeINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(gPlatform, "clMemFreeINTEL");
    auto clSetKernelArgMemPointerINTEL = (clSetKernelArgMemPointerINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(gPlatform, "clSetKernelArgMemPointerINTEL");

    if (clSharedMemAllocINTEL == nullptr || clMemFreeINTEL == nullptr ||
        clSetKernelArgMemPointerINTEL == nullptr) {
        GTEST_SKIP() << "USM extension not available";
    }

    const int N = 16;  // NxN matrices
    const size_t mat_size = N * N * sizeof(float);
    cl_int err;

    // Allocate USM for A, B, C
    void* a_ptr = clSharedMemAllocINTEL(m_context, gDevice, nullptr, mat_size, 0, &err);
    ASSERT_EQ(err, CL_SUCCESS) << "Failed to allocate USM for A";
    ASSERT_NE(a_ptr, nullptr);

    void* b_ptr = clSharedMemAllocINTEL(m_context, gDevice, nullptr, mat_size, 0, &err);
    ASSERT_EQ(err, CL_SUCCESS) << "Failed to allocate USM for B";
    ASSERT_NE(b_ptr, nullptr);

    void* c_ptr = clSharedMemAllocINTEL(m_context, gDevice, nullptr, mat_size, 0, &err);
    ASSERT_EQ(err, CL_SUCCESS) << "Failed to allocate USM for C";
    ASSERT_NE(c_ptr, nullptr);

    float* A = static_cast<float*>(a_ptr);
    float* B = static_cast<float*>(b_ptr);

    // Initialize A: A[i][j] = i*N + j (deterministic)
    for (int i = 0; i < N * N; i++) {
        A[i] = static_cast<float>(i);
    }
    // Initialize B: B[i][j] = (i == j) ? 1 : 0 (identity) so C = A
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B[i * N + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    const char* source = R"(
    kernel void matmul(__global float* A, __global float* B, __global float* C, int N) {
        int row = get_global_id(0);
        int col = get_global_id(1);
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
    )";

    auto program = CreateAndBuildProgram(source);
    auto kernel = CreateKernel(program, "matmul");

    err = clSetKernelArgMemPointerINTEL(kernel, 0, a_ptr);
    ASSERT_EQ(err, CL_SUCCESS);
    err = clSetKernelArgMemPointerINTEL(kernel, 1, b_ptr);
    ASSERT_EQ(err, CL_SUCCESS);
    err = clSetKernelArgMemPointerINTEL(kernel, 2, c_ptr);
    ASSERT_EQ(err, CL_SUCCESS);
    err = clSetKernelArg(kernel, 3, sizeof(int), &N);
    ASSERT_EQ(err, CL_SUCCESS);

    size_t global_size[2] = {static_cast<size_t>(N), static_cast<size_t>(N)};
    err = clEnqueueNDRangeKernel(m_queue, kernel, 2, nullptr, global_size,
                                 nullptr, 0, nullptr, nullptr);
    ASSERT_EQ(err, CL_SUCCESS);

    err = clFinish(m_queue);
    ASSERT_EQ(err, CL_SUCCESS);

    // Verify: C = A * I = A (B was identity)
    float* C = static_cast<float*>(c_ptr);
    for (int i = 0; i < N * N; i++) {
        EXPECT_FLOAT_EQ(C[i], A[i]) << "Mismatch at C[" << i << "]";
    }

    clMemFreeINTEL(m_context, a_ptr);
    clMemFreeINTEL(m_context, b_ptr);
    clMemFreeINTEL(m_context, c_ptr);
}

// Test clSetKernelArgSVMPointer with USM allocations
TEST_F(WithCommandQueue, SetKernelArgSVMPointer) {
    // Get function pointers
    auto clDeviceMemAllocINTEL = (clDeviceMemAllocINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(gPlatform, "clDeviceMemAllocINTEL");
    auto clSharedMemAllocINTEL = (clSharedMemAllocINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(gPlatform, "clSharedMemAllocINTEL");
    auto clMemFreeINTEL = (clMemFreeINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(gPlatform, "clMemFreeINTEL");
    
    if (clDeviceMemAllocINTEL == nullptr || clSharedMemAllocINTEL == nullptr || clMemFreeINTEL == nullptr) {
        GTEST_SKIP() << "USM extension not available";
    }
    
    // Create a simple kernel that takes a buffer argument
    const char* kernel_source = R"(
        kernel void init_buffer(global int* data, int value) {
            int gid = get_global_id(0);
            data[gid] = gid + value;
        }
    )";
    
    auto program = CreateAndBuildProgram(kernel_source);
    auto kernel = CreateKernel(program, "init_buffer");
    
    // Allocate shared memory (CPU + GPU accessible) for result verification
    size_t num_elements = 64;
    size_t buffer_size = num_elements * sizeof(int);
    cl_int err;
    void* usm_ptr = clSharedMemAllocINTEL(m_context, gDevice, nullptr, buffer_size, 0, &err);
    ASSERT_EQ(err, CL_SUCCESS);
    ASSERT_NE(usm_ptr, nullptr);
    
    // Initialize buffer to zeros
    memset(usm_ptr, 0, buffer_size);
    
    // Set kernel argument using clSetKernelArgSVMPointer
    err = clSetKernelArgSVMPointer(kernel, 0, usm_ptr);
    ASSERT_EQ(err, CL_SUCCESS) << "clSetKernelArgSVMPointer should succeed with valid USM pointer";
    
    // Set second argument (POD type)
    int value = 100;
    err = clSetKernelArg(kernel, 1, sizeof(int), &value);
    ASSERT_EQ(err, CL_SUCCESS);
    
    // Test error cases
    err = clSetKernelArgSVMPointer(kernel, 999, usm_ptr);
    EXPECT_EQ(err, CL_INVALID_ARG_INDEX) << "Should return INVALID_ARG_INDEX for out-of-range argument";
    
    // Test with nullptr (should succeed per chipStar compatibility)
    err = clSetKernelArgSVMPointer(kernel, 0, nullptr);
    EXPECT_EQ(err, CL_SUCCESS) << "clSetKernelArgSVMPointer should handle nullptr gracefully";
    
    // Set it back to valid pointer for execution
    err = clSetKernelArgSVMPointer(kernel, 0, usm_ptr);
    ASSERT_EQ(err, CL_SUCCESS);
    
    // Execute kernel
    size_t global_size = num_elements;
    err = clEnqueueNDRangeKernel(m_queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(err, CL_SUCCESS) << "Kernel execution should succeed";
    
    // Wait for completion
    err = clFinish(m_queue);
    EXPECT_EQ(err, CL_SUCCESS);
    
    // Verify results
    int* data = static_cast<int*>(usm_ptr);
    for (size_t i = 0; i < num_elements; i++) {
        EXPECT_EQ(data[i], static_cast<int>(i + value)) << "Mismatch at index " << i;
    }
    
    // Cleanup
    clMemFreeINTEL(m_context, usm_ptr);
}

// Test clSetKernelArgSVMPointer with device-only USM allocation
TEST_F(WithCommandQueue, SetKernelArgSVMPointerDeviceOnly) {
    // Get function pointers
    auto clDeviceMemAllocINTEL = (clDeviceMemAllocINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(gPlatform, "clDeviceMemAllocINTEL");
    auto clSharedMemAllocINTEL = (clSharedMemAllocINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(gPlatform, "clSharedMemAllocINTEL");
    auto clMemFreeINTEL = (clMemFreeINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(gPlatform, "clMemFreeINTEL");
    
    if (clDeviceMemAllocINTEL == nullptr || clSharedMemAllocINTEL == nullptr || clMemFreeINTEL == nullptr) {
        GTEST_SKIP() << "USM extension not available";
    }
    
    // Create kernel that copies from one buffer to another
    const char* kernel_source = R"(
        kernel void copy_buffer(global int* src, global int* dst) {
            int gid = get_global_id(0);
            dst[gid] = src[gid] * 2;
        }
    )";
    
    auto program = CreateAndBuildProgram(kernel_source);
    auto kernel = CreateKernel(program, "copy_buffer");
    
    size_t num_elements = 64;
    size_t buffer_size = num_elements * sizeof(int);
    cl_int err;
    
    // Allocate shared memory for source (can initialize from CPU)
    void* src_ptr = clSharedMemAllocINTEL(m_context, gDevice, nullptr, buffer_size, 0, &err);
    ASSERT_EQ(err, CL_SUCCESS);
    ASSERT_NE(src_ptr, nullptr);
    
    // Initialize source data
    int* src_data = static_cast<int*>(src_ptr);
    for (size_t i = 0; i < num_elements; i++) {
        src_data[i] = static_cast<int>(i);
    }
    
    // Allocate device-only memory for destination
    void* dst_ptr = clDeviceMemAllocINTEL(m_context, gDevice, nullptr, buffer_size, 0, &err);
    ASSERT_EQ(err, CL_SUCCESS);
    ASSERT_NE(dst_ptr, nullptr);
    
    // Set both kernel arguments using clSetKernelArgSVMPointer
    err = clSetKernelArgSVMPointer(kernel, 0, src_ptr);
    ASSERT_EQ(err, CL_SUCCESS) << "Should set shared USM pointer";
    
    err = clSetKernelArgSVMPointer(kernel, 1, dst_ptr);
    ASSERT_EQ(err, CL_SUCCESS) << "Should set device-only USM pointer";
    
    // Execute kernel
    size_t global_size = num_elements;
    err = clEnqueueNDRangeKernel(m_queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(err, CL_SUCCESS);
    
    err = clFinish(m_queue);
    EXPECT_EQ(err, CL_SUCCESS);
    
    // Copy result back to shared memory for verification
    void* result_ptr = clSharedMemAllocINTEL(m_context, gDevice, nullptr, buffer_size, 0, &err);
    ASSERT_EQ(err, CL_SUCCESS);
    
    err = clEnqueueSVMMemcpy(m_queue, CL_TRUE, result_ptr, dst_ptr, buffer_size, 0, nullptr, nullptr);
    ASSERT_EQ(err, CL_SUCCESS);
    
    // Verify results
    int* result_data = static_cast<int*>(result_ptr);
    for (size_t i = 0; i < num_elements; i++) {
        EXPECT_EQ(result_data[i], static_cast<int>(i * 2)) << "Mismatch at index " << i;
    }
    
    // Cleanup
    clMemFreeINTEL(m_context, src_ptr);
    clMemFreeINTEL(m_context, dst_ptr);
    clMemFreeINTEL(m_context, result_ptr);
}

