#ifndef VMM_PROJECT_SWAP_SWAP_MANAGER_H_
#define VMM_PROJECT_SWAP_SWAP_MANAGER_H_

#include <cstddef>
#include <string>

#include "swap/swap_context.h"
#include "vmm/vmm_allocator.h"

namespace vmm_project {

class SwapManager {
 public:
  explicit SwapManager(VmmAllocator* allocator);

  SwapManager(const SwapManager&) = delete;
  SwapManager& operator=(const SwapManager&) = delete;

  // Save all GPU allocations with the given tag to pinned host memory,
  // then release GPU physical backing. VA is preserved for VMM allocs.
  bool SwapOut(const std::string& tag);

  // Restore GPU allocations for the given tag from pinned host memory.
  bool SwapIn(const std::string& tag);

  // Number of contexts currently saved.
  std::size_t saved_count() const { return context_slot_; }

  // VRAM tracking.
  std::size_t vram_before_swap() const { return vram_before_; }
  std::size_t vram_after_swap() const { return vram_after_; }

 private:
  VmmAllocator* allocator_;
  std::string active_context_;
  SwapContext contexts_[2];
  int context_slot_ = 0;

  std::size_t vram_before_ = 0;
  std::size_t vram_after_ = 0;
};

}  // namespace vmm_project

#endif  // VMM_PROJECT_SWAP_SWAP_MANAGER_H_
