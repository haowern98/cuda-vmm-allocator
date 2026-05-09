#include "cuda/cuda_device.h"

#include <cuda.h>

#include "cuda/cuda_error.h"

namespace vmm_project {

int GetDeviceCount() {
  int count = 0;
  VMM_CUDA_CHECK(cuDeviceGetCount(&count));
  return count;
}

DeviceInfo GetDeviceInfo(int device_index) {
  CUdevice device;
  VMM_CUDA_CHECK(cuDeviceGet(&device, device_index));

  DeviceInfo info;
  info.index = device_index;

  char name_buf[256] = {};
  VMM_CUDA_CHECK(cuDeviceGetName(name_buf, sizeof(name_buf), device));
  info.name = name_buf;

  VMM_CUDA_CHECK(cuDeviceGetAttribute(
      &info.compute_capability_major,
      CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  VMM_CUDA_CHECK(cuDeviceGetAttribute(
      &info.compute_capability_minor,
      CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

  std::size_t mem_bytes = 0;
  VMM_CUDA_CHECK(cuDeviceTotalMem(&mem_bytes, device));
  info.total_memory_mib = mem_bytes / (1024 * 1024);

  int vmm_attr = 0;
  cuDeviceGetAttribute(&vmm_attr,
      CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device);
  info.vmm_supported = (vmm_attr != 0);

  int win32_attr = 0;
  cuDeviceGetAttribute(&win32_attr,
      CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED, device);
  info.win32_handle_supported = (win32_attr != 0);

  int win32_kmt_attr = 0;
  cuDeviceGetAttribute(&win32_kmt_attr,
      CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED, device);
  info.win32_kmt_handle_supported = (win32_kmt_attr != 0);

  int compression_attr = 0;
  cuDeviceGetAttribute(&compression_attr,
      CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, device);
  info.compression_supported = (compression_attr != 0);

  return info;
}

void SetDevice(int device_index) {
  CUdevice device;
  VMM_CUDA_CHECK(cuDeviceGet(&device, device_index));
  CUcontext ctx = nullptr;
  VMM_CUDA_CHECK(cuDevicePrimaryCtxRetain(&ctx, device));
  VMM_CUDA_CHECK(cuCtxSetCurrent(ctx));
}

}  // namespace vmm_project
