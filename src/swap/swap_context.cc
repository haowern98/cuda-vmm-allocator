#include "swap/swap_context.h"

#include <cuda.h>

#include "cuda/cuda_error.h"

namespace vmm_project {

SwapContext::~SwapContext() {
  if (host_buffer_) {
    VMM_CUDA_CHECK(cuMemFreeHost(host_buffer_));
  }
}

SwapContext::SwapContext(SwapContext&& other) noexcept
    : tag_(std::move(other.tag_)),
      entries_(std::move(other.entries_)),
      host_buffer_(other.host_buffer_),
      host_size_(other.host_size_) {
  other.host_buffer_ = nullptr;
  other.host_size_ = 0;
}

SwapContext& SwapContext::operator=(SwapContext&& other) noexcept {
  if (this != &other) {
    if (host_buffer_) cuMemFreeHost(host_buffer_);
    tag_ = std::move(other.tag_);
    entries_ = std::move(other.entries_);
    host_buffer_ = other.host_buffer_;
    host_size_ = other.host_size_;
    other.host_buffer_ = nullptr;
    other.host_size_ = 0;
  }
  return *this;
}

void SwapContext::Store(const std::string& tag,
                        const std::vector<SwapEntry>& entries,
                        void* host_buffer,
                        std::size_t host_size) {
  tag_ = tag;
  entries_ = entries;
  host_buffer_ = host_buffer;
  host_size_ = host_size;
}

}  // namespace vmm_project
