#include "vmm/vmm_allocator.h"

#include <cuda.h>

#include "cuda/cuda_device.h"
#include "cuda/cuda_error.h"

namespace vmm_project {

VmmAllocator::VmmAllocator() = default;

VmmAllocator::~VmmAllocator() {
  // Deallocate remaining entries. Iterate over a copy because
  // Deallocate modifies the map.
  std::vector<void*> ptrs;
  ptrs.reserve(entries_.size());
  for (const auto& [ptr, _] : entries_) {
    ptrs.push_back(ptr);
  }
  for (void* ptr : ptrs) {
    Deallocate(ptr);
  }
}

void VmmAllocator::Initialize(std::size_t device_index) {
  device_index_ = static_cast<int>(device_index);
  SetDevice(device_index_);

  // Query VMM granularity.
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = static_cast<int>(device_index);

  std::size_t min_granularity = 0;
  VMM_CUDA_CHECK(cuMemGetAllocationGranularity(
      &min_granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  granularity_ = min_granularity;

  pool_.Initialize(granularity_);
}

void* VmmAllocator::Allocate(std::size_t bytes, const char* tag) {
  SetDevice(device_index_);

  if (bytes >= granularity_) {
    return AllocateVmm(bytes, tag);
  }
  return AllocateFallback(bytes);
}

void VmmAllocator::Deallocate(void* ptr) {
  SetDevice(device_index_);

  auto it = entries_.find(ptr);
  if (it == entries_.end()) return;

  const Entry& entry = it->second;
  if (entry.is_vmm) {
    DeallocateVmm(entry);
  } else {
    DeallocateFallback(entry.ptr);
  }

  stats_.live_bytes -= entry.size;
  stats_.allocation_count -= 1;
  entries_.erase(it);
}

VmmAllocationInfo VmmAllocator::GetInfo(void* ptr) const {
  auto it = entries_.find(ptr);
  if (it == entries_.end()) {
    return {};
  }
  return {it->second.ptr, it->second.size, it->second.tag,
          it->second.is_vmm};
}

std::vector<VmmAllocationInfo> VmmAllocator::GetAllocationsByTag(
    const char* tag) const {
  std::vector<VmmAllocationInfo> result;
  for (const auto& [_, entry] : entries_) {
    if (entry.tag == tag) {
      result.push_back(
          {entry.ptr, entry.size, entry.tag, entry.is_vmm});
    }
  }
  return result;
}

void* VmmAllocator::AllocateVmm(std::size_t bytes, const char* tag) {
  // Round up to granularity.
  std::size_t alloc_size =
      ((bytes + granularity_ - 1) / granularity_) * granularity_;

  // Reserve virtual address space.
  void* va = pool_.AllocateVirtual(alloc_size);

  // Allocate physical memory.
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_index_;

  CUmemGenericAllocationHandle handle{};
  VMM_CUDA_CHECK(cuMemCreate(&handle, alloc_size, &prop, /*flags=*/0));

  // Map physical memory into the virtual address range.
  VMM_CUDA_CHECK(cuMemMap(reinterpret_cast<CUdeviceptr>(va),
                           alloc_size, /*offset=*/0, handle,
                           /*flags=*/0));

  // Set access: read-write.
  CUmemAccessDesc access_desc = {};
  access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access_desc.location.id = device_index_;
  access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  VMM_CUDA_CHECK(cuMemSetAccess(
      reinterpret_cast<CUdeviceptr>(va), alloc_size, &access_desc,
      /*count=*/1));

  Entry entry;
  entry.ptr = va;
  entry.size = bytes;  // Store requested size, not rounded
  entry.tag = tag;
  entry.is_vmm = true;
  entry.physical_handle = handle;
  entries_[va] = entry;

  stats_.live_bytes += bytes;
  stats_.peak_bytes = std::max(stats_.peak_bytes, stats_.live_bytes);
  stats_.allocation_count += 1;
  stats_.vmm_count += 1;

  return va;
}

void VmmAllocator::DeallocateVmm(const Entry& entry) {
  CUdeviceptr va = reinterpret_cast<CUdeviceptr>(entry.ptr);
  std::size_t alloc_size =
      ((entry.size + granularity_ - 1) / granularity_) * granularity_;

  // Unmap.
  cuMemUnmap(va, alloc_size);
  // Release physical memory.
  cuMemRelease(entry.physical_handle);
  // Free virtual address.
  pool_.FreeVirtual(entry.ptr, entry.size);
}

void* VmmAllocator::AllocateFallback(std::size_t bytes) {
  void* ptr = nullptr;
  CUresult result = cuMemAlloc(
      reinterpret_cast<CUdeviceptr*>(&ptr), bytes);
  if (result == CUDA_ERROR_OUT_OF_MEMORY) {
    return nullptr;
  }
  VMM_CUDA_CHECK(result);

  Entry entry;
  entry.ptr = ptr;
  entry.size = bytes;
  entry.tag = "fallback";
  entry.is_vmm = false;
  entry.physical_handle = CUmemGenericAllocationHandle{};
  entries_[ptr] = entry;

  stats_.live_bytes += bytes;
  stats_.peak_bytes = std::max(stats_.peak_bytes, stats_.live_bytes);
  stats_.allocation_count += 1;
  stats_.fallback_count += 1;

  return ptr;
}

void VmmAllocator::DeallocateFallback(void* ptr) {
  VMM_CUDA_CHECK(
      cuMemFree(reinterpret_cast<CUdeviceptr>(ptr)));
}

void VmmAllocator::ReleasePhysical(void* ptr) {
  auto it = entries_.find(ptr);
  if (it == entries_.end()) return;
  Entry& entry = it->second;
  if (!entry.is_vmm) return;

  CUdeviceptr va = reinterpret_cast<CUdeviceptr>(entry.ptr);
  std::size_t alloc_size =
      ((entry.size + granularity_ - 1) / granularity_) * granularity_;

  cuMemUnmap(va, alloc_size);
  cuMemRelease(entry.physical_handle);
  entry.physical_handle = CUmemGenericAllocationHandle{};

  stats_.live_bytes -= entry.size;
}

void VmmAllocator::RestorePhysical(void* ptr, std::size_t size,
                                   const void* host_data) {
  auto it = entries_.find(ptr);
  if (it == entries_.end()) return;
  Entry& entry = it->second;
  if (!entry.is_vmm) return;

  std::size_t alloc_size =
      ((entry.size + granularity_ - 1) / granularity_) * granularity_;

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_index_;

  CUmemGenericAllocationHandle handle{};
  VMM_CUDA_CHECK(cuMemCreate(&handle, alloc_size, &prop, /*flags=*/0));

  CUdeviceptr va = reinterpret_cast<CUdeviceptr>(entry.ptr);
  VMM_CUDA_CHECK(cuMemMap(va, alloc_size, /*offset=*/0, handle,
                           /*flags=*/0));

  CUmemAccessDesc access_desc = {};
  access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access_desc.location.id = device_index_;
  access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  VMM_CUDA_CHECK(cuMemSetAccess(va, alloc_size, &access_desc,
                                 /*count=*/1));

  VMM_CUDA_CHECK(cuMemcpyHtoD(va, host_data, entry.size));

  entry.physical_handle = handle;
  stats_.live_bytes += entry.size;
}

}  // namespace vmm_project

