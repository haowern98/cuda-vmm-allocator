#ifndef VMM_PROJECT_CUDA_CUDA_STREAM_H_
#define VMM_PROJECT_CUDA_CUDA_STREAM_H_

#include <cuda.h>

namespace vmm_project {

class CudaStream {
 public:
  CudaStream();
  ~CudaStream();

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  CudaStream(CudaStream&& other) noexcept;
  CudaStream& operator=(CudaStream&& other) noexcept;

  CUstream get() const { return stream_; }
  void Synchronize();

 private:
  CUstream stream_ = nullptr;
};

}  // namespace vmm_project

#endif  // VMM_PROJECT_CUDA_CUDA_STREAM_H_
