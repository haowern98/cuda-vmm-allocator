#ifndef VMM_PROJECT_INTERPOSE_INTERCEPT_TYPES_H_
#define VMM_PROJECT_INTERPOSE_INTERCEPT_TYPES_H_

// Common types shared by the interception layer.
// Declared manually to avoid cuda_runtime_api.h include dependencies.

#include <cstddef>

namespace vmm_project {

// Mirror of CUDA runtime types. Must match cudart.dll ABI.
using CudaRtError = int;
struct CudaRtDim3 { unsigned int x, y, z; };
using CudaRtStream = void*;

constexpr int kCudaSuccess = 0;
constexpr int kCudaErrorMemoryAllocation = 2;

}  // namespace vmm_project

#endif  // VMM_PROJECT_INTERPOSE_INTERCEPT_TYPES_H_
