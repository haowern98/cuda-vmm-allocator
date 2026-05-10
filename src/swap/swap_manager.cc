#include "swap/swap_manager.h"

#include <cstdio>

#include "vmm/vmm_allocator.h"
#include "vmm/vmm_types.h"

namespace vmm_project {

SwapManager::SwapManager(VmmAllocator* allocator)
    : allocator_(allocator) {}

bool SwapManager::SwapOut(const std::string& tag) {
  if (!allocator_) return false;

  vram_before_ = allocator_->stats().live_bytes;

  auto allocs = allocator_->GetAllocationsByTag(tag.c_str());
  if (allocs.empty()) {
    std::printf("SwapOut: no allocations for tag \"%s\"\n",
                tag.c_str());
    return false;
  }

  // Compute total host buffer size: metadata + aligned data per block.
  std::size_t meta_size = allocs.size() * sizeof(SwapEntry);
  std::size_t data_size = 0;
  for (const auto& a : allocs) data_size += a.size;
  // 256-byte align the data region.
  std::size_t data_offset =
      ((meta_size + 255) / 256) * 256;
  std::size_t total_size = data_offset + data_size;

  // Allocate pinned host memory.
  void* host_buf = nullptr;
  CUresult r = cuMemHostAlloc(&host_buf, total_size,
                               CU_MEMHOSTALLOC_PORTABLE);
  if (r != CUDA_SUCCESS) {
    std::printf("SwapOut: cuMemHostAlloc(%zu) failed\n",
                total_size);
    return false;
  }

  // Build metadata and copy data.
  auto* entries =
      static_cast<SwapEntry*>(host_buf);
  auto* data_region = static_cast<char*>(host_buf) + data_offset;
  std::size_t data_cursor = 0;

  std::vector<SwapEntry> entry_copy;
  entry_copy.reserve(allocs.size());

  for (const auto& a : allocs) {
    SwapEntry e{a.ptr, a.size, a.is_vmm};
    entries[entry_copy.size()] = e;
    entry_copy.push_back(e);

    // Copy GPU data to pinned host.
    cuMemcpyDtoH(data_region + data_cursor,
                 reinterpret_cast<CUdeviceptr>(a.ptr), a.size);
    data_cursor += a.size;
  }

  // Release GPU physical backing.
  for (const auto& a : allocs) {
    if (a.is_vmm) {
      allocator_->ReleasePhysical(a.ptr);
    } else {
      allocator_->Deallocate(a.ptr);
    }
  }

  // Store context.
  int slot = context_slot_++;
  contexts_[slot].Store(tag, entry_copy, host_buf, total_size);

  vram_after_ = allocator_->stats().live_bytes;

  std::printf("SwapOut: \"%s\" saved %zu bytes to host, "
              "VRAM %zu -> %zu KiB\n",
              tag.c_str(), total_size,
              vram_before_ / 1024, vram_after_ / 1024);
  return true;
}

bool SwapManager::SwapIn(const std::string& tag) {
  if (!allocator_) return false;

  // Find context for this tag.
  SwapContext* ctx = nullptr;
  for (int i = 0; i < context_slot_; ++i) {
    if (contexts_[i].tag() == tag) {
      ctx = &contexts_[i];
      break;
    }
  }
  if (!ctx || !ctx->valid()) {
    std::printf("SwapIn: no saved context for \"%s\"\n",
                tag.c_str());
    return false;
  }

  vram_before_ = allocator_->stats().live_bytes;

  const auto& entries = ctx->entries();
  const void* host_data = ctx->host_data();

  // Data region offset.
  std::size_t meta_size = entries.size() * sizeof(SwapEntry);
  std::size_t data_offset =
      ((meta_size + 255) / 256) * 256;
  auto* data_region =
      static_cast<const char*>(host_data) + data_offset;
  std::size_t data_cursor = 0;

  for (const auto& e : entries) {
    if (e.is_vmm) {
      allocator_->RestorePhysical(e.original_ptr, e.size,
                                  data_region + data_cursor);
    } else {
      // Re-allocate fallback and copy data.
      void* ptr = allocator_->Allocate(e.size, tag.c_str());
      if (ptr) {
        cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(ptr),
                     data_region + data_cursor, e.size);
      }
    }
    data_cursor += e.size;
  }

  vram_after_ = allocator_->stats().live_bytes;

  std::printf("SwapIn: \"%s\" restored, VRAM %zu -> %zu KiB\n",
              tag.c_str(), vram_before_ / 1024,
              vram_after_ / 1024);
  return true;
}

}  // namespace vmm_project
