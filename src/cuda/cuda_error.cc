#include "cuda/cuda_error.h"

#include <cuda.h>
#include <sstream>

namespace vmm_project {

CudaError::CudaError(CUresult result, const char* file, int line,
                     const char* expr)
    : code_(result) {
  const char* name = nullptr;
  const char* desc = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &desc);
  std::ostringstream oss;
  oss << file << ":" << line << ": CUDA error at " << expr << " -- "
      << (name ? name : "UNKNOWN") << ": "
      << (desc ? desc : "no description available");
  message_ = oss.str();
}

const char* CudaError::what() const noexcept {
  return message_.c_str();
}

void CudaCheck(CUresult result, const char* file, int line,
               const char* expr) {
  if (result != CUDA_SUCCESS) {
    throw CudaError(result, file, line, expr);
  }
}

}  // namespace vmm_project
