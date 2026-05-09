#ifndef VMM_PROJECT_VMM_VMM_TYPES_H_
#define VMM_PROJECT_VMM_VMM_TYPES_H_

#include <cstddef>
#include <cstdint>
#include <string>

namespace vmm_project {

struct VmmAllocationInfo {
  void* ptr = nullptr;
  std::size_t size = 0;
  std::string tag;
  bool is_vmm = false;
};

struct VmmStats {
  std::size_t live_bytes = 0;
  std::size_t peak_bytes = 0;
  std::size_t allocation_count = 0;
  std::size_t vmm_count = 0;
  std::size_t fallback_count = 0;
};

}  // namespace vmm_project

#endif  // VMM_PROJECT_VMM_VMM_TYPES_H_
