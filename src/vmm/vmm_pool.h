#ifndef VMM_PROJECT_VMM_VMM_POOL_H_
#define VMM_PROJECT_VMM_VMM_POOL_H_

#include <cstddef>
#include <map>
#include <vector>

namespace vmm_project {

// Manages GPU virtual address space for VMM allocations.
// Reserves VA in chunks and hands out aligned sub-ranges.
class VmmPool {
 public:
  VmmPool();
  ~VmmPool();

  VmmPool(const VmmPool&) = delete;
  VmmPool& operator=(const VmmPool&) = delete;

  // Initialize with the minimum granularity from the device.
  void Initialize(std::size_t granularity);

  // Reserve a VA range of the given size. Returns the base pointer.
  void* AllocateVirtual(std::size_t size);

  // Free a previously allocated VA range.
  void FreeVirtual(void* ptr, std::size_t size);

  std::size_t granularity() const { return granularity_; }

 private:
  // VA range [start, end) available for allocation.
  struct FreeRange {
    std::uintptr_t start;
    std::uintptr_t end;
  };

  void ReserveChunk();

  std::size_t granularity_ = 0;
  // Sorted by start address.
  std::vector<FreeRange> free_ranges_;
  // Chunk size for new VA reservations (256 MB).
  static constexpr std::size_t kChunkSize = 256 * 1024 * 1024;
};

}  // namespace vmm_project

#endif  // VMM_PROJECT_VMM_VMM_POOL_H_
