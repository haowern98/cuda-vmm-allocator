#ifndef VMM_PROJECT_SWAP_SWAP_CONTEXT_H_
#define VMM_PROJECT_SWAP_SWAP_CONTEXT_H_

#include <cstddef>
#include <string>
#include <vector>

namespace vmm_project {

struct SwapEntry {
  void* original_ptr;
  std::size_t size;
  bool is_vmm;
};

class SwapContext {
 public:
  SwapContext() = default;
  ~SwapContext();

  SwapContext(const SwapContext&) = delete;
  SwapContext& operator=(const SwapContext&) = delete;

  SwapContext(SwapContext&& other) noexcept;
  SwapContext& operator=(SwapContext&& other) noexcept;

  // Store metadata and take ownership of a pinned host buffer.
  void Store(const std::string& tag,
             const std::vector<SwapEntry>& entries,
             void* host_buffer, std::size_t host_size);

  // Accessors.
  const std::string& tag() const { return tag_; }
  const std::vector<SwapEntry>& entries() const { return entries_; }
  const void* host_data() const { return host_buffer_; }
  std::size_t host_size() const { return host_size_; }
  bool valid() const { return host_buffer_ != nullptr; }

 private:
  std::string tag_;
  std::vector<SwapEntry> entries_;
  void* host_buffer_ = nullptr;
  std::size_t host_size_ = 0;
};

}  // namespace vmm_project

#endif  // VMM_PROJECT_SWAP_SWAP_CONTEXT_H_
