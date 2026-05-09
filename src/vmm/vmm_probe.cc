#include "vmm/vmm_probe.h"

#include <cstdio>
#include <cuda.h>

#include "cuda/cuda_device.h"
#include "cuda/cuda_error.h"

namespace vmm_project {

namespace {

void PrintGranularity(const char* label,
                      CUmemAllocationGranularity_flags option) {
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;

  std::size_t granularity = 0;
  CUresult result = cuMemGetAllocationGranularity(&granularity, &prop,
                                                   option);
  if (result == CUDA_SUCCESS) {
    std::printf("  %s Granularity: %zu bytes (%zu KiB)\n",
                label, granularity, granularity / 1024);
  } else {
    const char* err_name = nullptr;
    cuGetErrorName(result, &err_name);
    std::printf("  %s Granularity: query failed (%s)\n", label,
                err_name ? err_name : "unknown");
  }
}

}  // namespace

int RunProbe() {
  // Initialize CUDA.
  CUresult init_result = cuInit(0);
  if (init_result != CUDA_SUCCESS) {
    std::printf("cuInit failed — no CUDA driver?\n");
    return 1;
  }

  int device_count = GetDeviceCount();
  std::printf("CUDA devices found: %d\n\n", device_count);

  if (device_count == 0) {
    std::printf("No CUDA-capable devices detected.\n");
    return 1;
  }

  // Probe each device for VMM support.
  bool any_vmm = false;
  for (int i = 0; i < device_count; ++i) {
    DeviceInfo info = GetDeviceInfo(i);
    std::printf("Device %d: %s\n", info.index, info.name.c_str());
    std::printf("  Compute Capability: %d.%d\n",
                info.compute_capability_major,
                info.compute_capability_minor);
    std::printf("  Total Memory: %zu MiB\n", info.total_memory_mib);
    std::printf("  Virtual Memory Management: %s\n",
                info.vmm_supported ? "YES" : "NO");
    std::printf("  Win32 Handle Export: %s\n",
                info.win32_handle_supported ? "YES" : "NO");
    std::printf("  Win32 KMT Handle Export: %s\n",
                info.win32_kmt_handle_supported ? "YES" : "NO");
    std::printf("  Compression: %s\n",
                info.compression_supported ? "YES" : "NO");

    if (info.vmm_supported) {
      any_vmm = true;
      SetDevice(i);

      PrintGranularity("Minimum",
                       CU_MEM_ALLOC_GRANULARITY_MINIMUM);
      PrintGranularity("Recommended",
                       CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);

      CUcontext ctx;
      cuCtxGetCurrent(&ctx);
      if (ctx) cuCtxDestroy(ctx);
    }
    std::printf("\n");
  }

  if (!any_vmm) {
    std::printf(
        "RESULT: No VMM-capable GPU found. "
        "The project cannot proceed as designed.\n");
    return 1;
  }

  std::printf("RESULT: VMM is supported. Project can proceed.\n");
  return 0;
}

}  // namespace vmm_project
