#ifndef VMM_PROJECT_CUDA_CUDA_ERROR_H_
#define VMM_PROJECT_CUDA_CUDA_ERROR_H_

#include <cuda.h>
#include <exception>
#include <string>

namespace vmm_project {

class CudaError : public std::exception {
 public:
  CudaError(CUresult result, const char* file, int line,
            const char* expr);
  const char* what() const noexcept override;
  CUresult code() const { return code_; }

 private:
  CUresult code_;
  std::string message_;
};

void CudaCheck(CUresult result, const char* file, int line,
               const char* expr);

}  // namespace vmm_project

#define VMM_CUDA_CHECK(expr) \
  vmm_project::CudaCheck((expr), __FILE__, __LINE__, #expr)

#endif  // VMM_PROJECT_CUDA_CUDA_ERROR_H_
