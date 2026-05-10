cmake_minimum_required(VERSION 3.26)
project(vmm-project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# --- Backend options ---
option(VMM_ENABLE_LLAMA "Enable llama.cpp backend runner" OFF)
option(VMM_ENABLE_ONNX  "Enable ONNX Runtime backend runner" OFF)

# --- Optional llama.cpp ---
if(VMM_ENABLE_LLAMA)
  set(VMM_LLAMA_DIR "${CMAKE_SOURCE_DIR}/../llama.cpp"
      CACHE PATH "Path to llama.cpp source")
  set(VMM_LLAMA_BUILD_DIR "${VMM_LLAMA_DIR}/build-cuda"
      CACHE PATH "Path to llama.cpp build directory")
  if(NOT EXISTS "${VMM_LLAMA_DIR}/include/llama.h")
    message(FATAL_ERROR "llama.h not found at ${VMM_LLAMA_DIR}/include/llama.h")
  endif()
  message(STATUS "llama.cpp runner: ENABLED")
  message(STATUS "  source: ${VMM_LLAMA_DIR}")
  message(STATUS "  build:  ${VMM_LLAMA_BUILD_DIR}")
endif()

# --- Optional ONNX Runtime ---
if(VMM_ENABLE_ONNX)
  set(VMM_ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/../mosaicvram/deps/onnxruntime-win-x64-gpu-1.25.1"
      CACHE PATH "Path to ONNX Runtime GPU SDK")
  if(NOT EXISTS "${VMM_ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "onnxruntime_cxx_api.h not found at ${VMM_ONNXRUNTIME_DIR}/include/")
  endif()
  message(STATUS "ONNX Runtime runner: ENABLED")
  message(STATUS "  sdk: ${VMM_ONNXRUNTIME_DIR}")
endif()

# --- Source files ---
set(VMM_SOURCES
    src/main.cc
    src/cuda/cuda_error.cc
    src/cuda/cuda_device.cc
    src/cuda/cuda_stream.cc
    src/vmm/vmm_probe.cc
    src/vmm/vmm_pool.cc
    src/vmm/vmm_allocator.cc
    src/interpose/intercept_real.cc
    src/interpose/intercept_hook.cc
    src/swap/swap_context.cc
    src/swap/swap_manager.cc
)

set(VMM_HEADERS
    src/cuda/cuda_error.h
    src/cuda/cuda_device.h
    src/cuda/cuda_stream.h
    src/util/timer.h
    src/backend/backend_runner.h
    src/vmm/vmm_probe.h
    src/vmm/vmm_types.h
    src/vmm/vmm_pool.h
    src/vmm/vmm_allocator.h
    src/interpose/intercept_real.h
    src/interpose/intercept_hook.h
    src/interpose/intercept_types.h
    src/swap/swap_context.h
    src/swap/swap_manager.h
)

if(VMM_ENABLE_LLAMA)
  list(APPEND VMM_SOURCES src/backend/llama_runner.cc)
  list(APPEND VMM_HEADERS src/backend/llama_runner.h)
endif()

if(VMM_ENABLE_ONNX)
  list(APPEND VMM_SOURCES src/backend/onnx_runner.cc)
  list(APPEND VMM_HEADERS src/backend/onnx_runner.h)
endif()

# --- Executable ---
add_executable(vmm-project ${VMM_SOURCES} ${VMM_HEADERS})

target_include_directories(vmm-project PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(vmm-project PRIVATE
    ${CUDAToolkit_LIBRARY_DIR}/cuda.lib
)

if(VMM_ENABLE_LLAMA)
  target_include_directories(vmm-project PRIVATE
      "${VMM_LLAMA_DIR}/include"
      "${VMM_LLAMA_DIR}/ggml/include"
      "${VMM_LLAMA_DIR}/ggml/src"
  )
  target_link_libraries(vmm-project PRIVATE
      "${VMM_LLAMA_BUILD_DIR}/src/llama.lib"
  )
  target_compile_definitions(vmm-project PRIVATE VMM_ENABLE_LLAMA)
endif()

if(VMM_ENABLE_ONNX)
  target_include_directories(vmm-project PRIVATE
      "${VMM_ONNXRUNTIME_DIR}/include"
  )
  target_link_libraries(vmm-project PRIVATE
      "${VMM_ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
  )
  target_compile_definitions(vmm-project PRIVATE VMM_ENABLE_ONNX)
endif()

set_target_properties(vmm-project PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
