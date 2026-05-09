#ifndef VMM_PROJECT_CUDA_CUDA_DEVICE_H_
#define VMM_PROJECT_CUDA_CUDA_DEVICE_H_

#include <cstddef>
#include <cstdint>
#include <string>

namespace vmm_project {

struct DeviceInfo {
  int index = 0;
  std::string name;
  int compute_capability_major = 0;
  int compute_capability_minor = 0;
  std::size_t total_memory_mib = 0;
  bool vmm_supported = false;
  bool win32_handle_supported = false;
  bool win32_kmt_handle_supported = false;
  bool compression_supported = false;
};

int GetDeviceCount();

DeviceInfo GetDeviceInfo(int device_index);

void SetDevice(int device_index);

}  // namespace vmm_project

#endif  // VMM_PROJECT_CUDA_CUDA_DEVICE_H_
