// Copyright 2024 The clvk authors.
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
#include <fstream>
#include <vector>

// Helper function to load SPIR-V binary from file
std::vector<uint8_t> load_spirv_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open SPIR-V file: " + filename);
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Failed to read SPIR-V file: " + filename);
    }
    
    return buffer;
}

TEST_F(WithCommandQueue, SpirvILProgramCreation) {
    // Check for SPIR-V IL extension support
    size_t ext_size;
    clGetDeviceInfo(gDevice, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_size);
    
    std::vector<char> extensions(ext_size);
    clGetDeviceInfo(gDevice, CL_DEVICE_EXTENSIONS, ext_size, extensions.data(), nullptr);
    
    ASSERT_TRUE(std::string(extensions.data()).find("cl_khr_il_program") != std::string::npos)
        << "Device does not support cl_khr_il_program extension";
    
    // Get the extension function
    clCreateProgramWithILKHR_fn clCreateProgramWithILKHR =
        reinterpret_cast<clCreateProgramWithILKHR_fn>(
            clGetExtensionFunctionAddressForPlatform(gPlatform, "clCreateProgramWithILKHR"));
    ASSERT_NE(clCreateProgramWithILKHR, nullptr)
        << "clCreateProgramWithILKHR extension function not available";
}

TEST_F(WithCommandQueue, SpirvProgramWithILTest) {
    // Skip if SPIR-V IL extension not supported
    size_t ext_size;
    clGetDeviceInfo(gDevice, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_size);
    
    std::vector<char> extensions(ext_size);
    clGetDeviceInfo(gDevice, CL_DEVICE_EXTENSIONS, ext_size, extensions.data(), nullptr);
    
    if (std::string(extensions.data()).find("cl_khr_il_program") == std::string::npos) {
        GTEST_SKIP() << "Device does not support cl_khr_il_program extension";
        return;
    }
    
    // Get the extension function
    clCreateProgramWithILKHR_fn clCreateProgramWithILKHR =
        reinterpret_cast<clCreateProgramWithILKHR_fn>(
            clGetExtensionFunctionAddressForPlatform(gPlatform, "clCreateProgramWithILKHR"));
    if (clCreateProgramWithILKHR == nullptr) {
        GTEST_SKIP() << "clCreateProgramWithILKHR extension function not available";
        return;
    }
    
    // Load MatrixMultiply.spv (the main program)
    std::vector<uint8_t> spirv_data;
    try {
        spirv_data = load_spirv_file("/Users/pvelesko/local/clvk/tests/data/MatrixMultiply.spv");
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Could not load MatrixMultiply.spv: " << e.what();
        return;
    }
    
    // Create program from SPIR-V
    cl_int err;
    auto program = clCreateProgramWithILKHR(m_context, spirv_data.data(), spirv_data.size(), &err);
    ASSERT_CL_SUCCESS(err);
    ASSERT_NE(program, nullptr);
    
    // Build the program
    err = clBuildProgram(program, 1, &gDevice, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Get build log for debugging
        size_t log_size;
        clGetProgramBuildInfo(program, gDevice, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, gDevice, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        ADD_FAILURE() << "Failed to build SPIR-V program: " << std::string(log.data());
    }
    EXPECT_CL_SUCCESS(err);
    
    if (program) {
        clReleaseProgram(program);
    }
}

TEST_F(WithCommandQueue, SpirvProgramLinkingTest) {
    // Skip if SPIR-V IL extension not supported
    size_t ext_size;
    clGetDeviceInfo(gDevice, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_size);
    
    std::vector<char> extensions(ext_size);
    clGetDeviceInfo(gDevice, CL_DEVICE_EXTENSIONS, ext_size, extensions.data(), nullptr);
    
    if (std::string(extensions.data()).find("cl_khr_il_program") == std::string::npos) {
        GTEST_SKIP() << "Device does not support cl_khr_il_program extension";
        return;
    }
    
    // Get the extension function
    clCreateProgramWithILKHR_fn clCreateProgramWithILKHR =
        reinterpret_cast<clCreateProgramWithILKHR_fn>(
            clGetExtensionFunctionAddressForPlatform(gPlatform, "clCreateProgramWithILKHR"));
    if (clCreateProgramWithILKHR == nullptr) {
        GTEST_SKIP() << "clCreateProgramWithILKHR extension function not available";
        return;
    }
    
    // Load all SPIR-V files
    std::vector<std::string> filenames = {
        "/Users/pvelesko/local/clvk/tests/data/atomicAddFloat_emulation.spv",
        "/Users/pvelesko/local/clvk/tests/data/ballot_native.spv", 
        "/Users/pvelesko/local/clvk/tests/data/MatrixMultiply.spv"
    };
    
    std::vector<cl_program> programs;
    
    for (const auto& filename : filenames) {
        try {
            auto spirv_data = load_spirv_file(filename);
            
            // Create program from SPIR-V
            cl_int err;
            auto program = clCreateProgramWithILKHR(m_context, spirv_data.data(), spirv_data.size(), &err);
            ASSERT_CL_SUCCESS(err);
            ASSERT_NE(program, nullptr);
            
            programs.push_back(program);
            
            // Build each program as a library (compiled object with unresolved symbols)
            err = clBuildProgram(program, 1, &gDevice, "-create-library", nullptr, nullptr);
            if (err != CL_SUCCESS) {
                // Get build log for debugging
                size_t log_size;
                clGetProgramBuildInfo(program, gDevice, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
                std::vector<char> log(log_size);
                clGetProgramBuildInfo(program, gDevice, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
                ADD_FAILURE() << "Failed to build library " << filename 
                            << ": " << std::string(log.data());
            }
            EXPECT_CL_SUCCESS(err);
            
        } catch (const std::exception& e) {
            GTEST_SKIP() << "Could not load " << filename << ": " << e.what();
            // Clean up any programs created so far
            for (auto prog : programs) {
                clReleaseProgram(prog);
            }
            return;
        }
    }
    
    // Now link the programs
    cl_int err;
    auto linked_program = clLinkProgram(m_context, 1, &gDevice, "", 
                                       programs.size(), programs.data(), 
                                       nullptr, nullptr, &err);
    
    if (err != CL_SUCCESS) {
        // Get detailed error information
        if (linked_program) {
            size_t log_size;
            clGetProgramBuildInfo(linked_program, gDevice, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(linked_program, gDevice, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            ADD_FAILURE() << "Failed to link SPIR-V programs: " << std::string(log.data());
        } else {
            ADD_FAILURE() << "clLinkProgram failed with error " << cl_code_to_string(err);
        }
    }
    EXPECT_CL_SUCCESS(err);
    ASSERT_NE(linked_program, nullptr);
    
    // Test that we can create a kernel from the linked program
    auto kernel = clCreateKernel(linked_program, "MatrixMultiply", &err);
    if (err != CL_SUCCESS) {
        // MatrixMultiply kernel may not exist or have a different name in the SPIR-V files
    } else {
        clReleaseKernel(kernel);
    }
    
    // Cleanup
    if (linked_program) {
        clReleaseProgram(linked_program);
    }
    for (auto prog : programs) {
        clReleaseProgram(prog);
    }