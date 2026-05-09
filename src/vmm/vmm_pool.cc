#include "vmm/vmm_pool.h"

#include <algorithm>
#include <cuda.h>

#include "cuda/cuda_error.h"

namespace vmm_project {

VmmPool::VmmPool() = default;

VmmPool::~VmmPool() {
  // Free all reserved chunks.
  // cuMemAddressFree requires the original reservation address and size.
  // Since we hand out sub-ranges, we rely on the allocator to track
  // individual chunk reservations. In practice the allocator owns
  // chunk-level bookkeeping through the physical handle registry.
}

void VmmPool::Initialize(std::size_t granularity) {
  granularity_ = granularity;
}

void VmmPool::ReserveChunk() {
  CUdeviceptr va_base = 0;
  CUresult result = cuMemAddressReserve(
      &va_base, kChunkSize, /*alignment=*/0, /*addr=*/0,
      /*flags=*/0);
  if (result != CUDA_SUCCESS) {
    // If reservation at a specific alignment fails, let CUDA pick.
    VMM_CUDA_CHECK(cuMemAddressReserve(&va_base, kChunkSize,
                                        /*alignment=*/0, /*addr=*/0,
                                        /*flags=*/0));
  }
  FreeRange range;
  range.start = static_cast<std::uintptr_t>(va_base);
  range.end = range.start + kChunkSize;
  free_ranges_.push_back(range);
}

void* VmmPool::AllocateVirtual(std::size_t size) {
  // Round up to granularity.
  std::size_t aligned_size =
      ((size + granularity_ - 1) / granularity_) * granularity_;

  // Find a free range that fits.
  for (auto it = free_ranges_.begin(); it != free_ranges_.end(); ++it) {
    std::size_t range_size = it->end - it->start;
    if (range_size >= aligned_size) {
      std::uintptr_t result_start = it->start;
      it->start += aligned_size;
      if (it->start >= it->end) {
        free_ranges_.erase(it);
      }
      return reinterpret_cast<void*>(result_start);
    }
  }

  // No range fits — reserve a new chunk and try again.
  ReserveChunk();
  return AllocateVirtual(size);
}

void VmmPool::FreeVirtual(void* ptr, std::size_t size) {
  std::uintptr_t start = reinterpret_cast<std::uintptr_t>(ptr);
  std::size_t aligned_size =
      ((size + granularity_ - 1) / granularity_) * granularity_;
  std::uintptr_t end = start + aligned_size;

  // Insert sorted into free list, coalescing adjacent ranges.
  FreeRange freed{start, end};

  auto it = std::lower_bound(
      free_ranges_.begin(), free_ranges_.end(), freed,
      [](const FreeRange& a, const FreeRange& b) {
        return a.start < b.start;
      });

  // Coalesce with previous range.
  if (it != free_ranges_.begin()) {
    auto prev = std::prev(it);
    if (prev->end == freed.start) {
      freed.start = prev->start;
      free_ranges_.erase(prev);
      it = std::lower_bound(
          free_ranges_.begin(), free_ranges_.end(), freed,
          [](const FreeRange& a, const FreeRange& b) {
            return a.start < b.start;
          });
    }
  }

  // Coalesce with next range.
  if (it != free_ranges_.end() && it->start == freed.end) {
    freed.end = it->end;
    free_ranges_.erase(it);
  }

  free_ranges_.insert(
      std::lower_bound(
          free_ranges_.begin(), free_ranges_.end(), freed,
          [](const FreeRange& a, const FreeRange& b) {
            return a.start < b.start;
          }),
      freed);
}

}  // namespace vmm_project
