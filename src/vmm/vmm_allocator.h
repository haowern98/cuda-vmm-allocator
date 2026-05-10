#ifndef VMM_PROJECT_VMM_VMM_ALLOCATOR_H_
#define VMM_PROJECT_VMM_VMM_ALLOCATOR_H_

#include <cstddef>
#include <cuda.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "vmm/vmm_pool.h"
#include "vmm/vmm_types.h"

namespace vmm_project {

class VmmAllocator {
 public:
  VmmAllocator();
  ~VmmAllocator();

  VmmAllocator(const VmmAllocator&) = delete;
  VmmAllocator& operator=(const VmmAllocator&) = delete;

  void Initialize(std::size_t device_index);

  void* Allocate(std::size_t bytes, const char* tag);
  void Deallocate(void* ptr);

  VmmAllocationInfo GetInfo(void* ptr) const;
  std::vector<VmmAllocationInfo> GetAllocationsByTag(
      const char* tag) const;
  VmmStats stats() const { return stats_; }

  // Swap support: release GPU physical backing, keep VA reserved.
  void ReleasePhysical(void* ptr);

  // Swap support: restore GPU physical backing at existing VA.
  void RestorePhysical(void* ptr, std::size_t size,
                       const void* host_data);

 private:
  struct Entry {
    void* ptr;
    std::size_t size;
    std::string tag;
    bool is_vmm;
    // Only valid when is_vmm is true.
    CUmemGenericAllocationHandle physical_handle;
  };

  void* AllocateVmm(std::size_t bytes, const char* tag);
  void DeallocateVmm(const Entry& entry);
  void* AllocateFallback(std::size_t bytes);
  void DeallocateFallback(void* ptr);

  VmmPool pool_;
  std::size_t granularity_ = 0;
  // Only valid on first device; context must be current during calls.
  int device_index_ = 0;

  std::unordered_map<void*, Entry> entries_;
  VmmStats stats_;
};

}  // namespace vmm_project

#endif  // VMM_PROJECT_VMM_VMM_ALLOCATOR_H_
