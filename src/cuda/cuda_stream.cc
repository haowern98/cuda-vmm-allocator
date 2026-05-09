#include "cuda/cuda_stream.h"

#include <utility>

#include "cuda/cuda_error.h"

namespace vmm_project {

CudaStream::CudaStream() {
  VMM_CUDA_CHECK(
      cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING));
}

CudaStream::~CudaStream() {
  if (stream_) {
    cuStreamDestroy(stream_);
  }
}

CudaStream::CudaStream(CudaStream&& other) noexcept
    : stream_(other.stream_) {
  other.stream_ = nullptr;
}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
  if (this != &other) {
    if (stream_) {
      cuStreamDestroy(stream_);
    }
    stream_ = other.stream_;
    other.stream_ = nullptr;
  }
  return *this;
}

void CudaStream::Synchronize() {
  VMM_CUDA_CHECK(cuStreamSynchronize(stream_));
}

}  // namespace vmm_project
