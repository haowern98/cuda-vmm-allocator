cmake_minimum_required(VERSION 3.26)
project(vmm-project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(VMM_SOURCES
    src/main.cc
    src/cuda/cuda_error.cc
    src/cuda/cuda_device.cc
    src/cuda/cuda_stream.cc
    src/vmm/vmm_probe.cc
    src/vmm/vmm_pool.cc
    src/vmm/vmm_allocator.cc
)

set(VMM_HEADERS
    src/cuda/cuda_error.h
    src/cuda/cuda_device.h
    src/cuda/cuda_stream.h
    src/util/timer.h
    src/vmm/vmm_probe.h
    src/vmm/vmm_types.h
    src/vmm/vmm_pool.h
    src/vmm/vmm_allocator.h
)

add_executable(vmm-project ${VMM_SOURCES} ${VMM_HEADERS})

target_include_directories(vmm-project PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(vmm-project PRIVATE
    ${CUDAToolkit_LIBRARY_DIR}/cuda.lib
)

set_target_properties(vmm-project PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
