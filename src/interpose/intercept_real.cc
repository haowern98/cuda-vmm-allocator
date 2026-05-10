#include "interpose/intercept_real.h"

#include <cstddef>
#include <cstdio>

#ifdef _WIN32
#include <windows.h>
#else
#error "intercept_real requires Windows"
#endif

#include <cuda.h>

#include "interpose/intercept_hook.h"
#include "vmm/vmm_allocator.h"

namespace vmm_project {

namespace {

// CUDA runtime types (ABI-compatible with cudart.dll).
using CudaError = int;
using CudaStream = void*;
struct Dim3 { unsigned int x, y, z; };
enum { kCudaSuccess = 0,
       kCudaErrorMemoryAlloc = 2,
       kMemcpyHostToDevice = 1,
       kMemcpyDeviceToHost = 2,
       kMemcpyDeviceToDevice = 3 };

// Function pointer types.
using MallocFn = CudaError (*)(void**, size_t);
using FreeFn = CudaError (*)(void*);
using MemcpyFn = CudaError (*)(void*, const void*, size_t, int);
using LaunchFn = CudaError (*)(const void*, Dim3, Dim3, void**,
                               size_t, CudaStream);

// Resolved from cudart64_12.dll.
HMODULE g_cuda_dll = nullptr;
MallocFn g_real_malloc = nullptr;
FreeFn g_real_free = nullptr;
MemcpyFn g_real_memcpy = nullptr;
LaunchFn g_real_launch = nullptr;

// Trampolines.
MallocFn g_tramp_malloc = nullptr;
FreeFn g_tramp_free = nullptr;
MemcpyFn g_tramp_memcpy = nullptr;
LaunchFn g_tramp_launch = nullptr;
void* g_raw_tramp_malloc = nullptr;
void* g_raw_tramp_free = nullptr;
void* g_raw_tramp_memcpy = nullptr;
void* g_raw_tramp_launch = nullptr;

// Counters.
std::size_t g_alloc_count = 0;
std::size_t g_free_count = 0;
std::size_t g_memcpy_count = 0;
std::size_t g_launch_count = 0;

// VMM allocator (lazy-init).
VmmAllocator g_vmm_allocator;
bool g_vmm_active = false;

// --- Hook implementations ---

CudaError Hook_Malloc(void** devPtr, size_t size) {
  ++g_alloc_count;
  if (g_vmm_active) {
    void* ptr = g_vmm_allocator.Allocate(size, "cuda");
    if (!ptr) return kCudaErrorMemoryAlloc;
    *devPtr = ptr;
    return kCudaSuccess;
  }
  return g_tramp_malloc(devPtr, size);
}

CudaError Hook_Free(void* devPtr) {
  ++g_free_count;
  if (g_vmm_active && devPtr) {
    g_vmm_allocator.Deallocate(devPtr);
    return kCudaSuccess;
  }
  // Cannot safely trampoline cudaFree — use driver API.
  return static_cast<CudaError>(
      cuMemFree(reinterpret_cast<CUdeviceptr>(devPtr)));
}

CudaError Hook_Memcpy(void* dst, const void* src, size_t count,
                      int kind) {
  ++g_memcpy_count;
  return g_tramp_memcpy(dst, src, count, kind);
}

CudaError Hook_Launch(const void* func, Dim3 grid, Dim3 block,
                      void** args, size_t shared,
                      CudaStream stream) {
  ++g_launch_count;
  return g_tramp_launch(func, grid, block, args, shared, stream);
}

// --- Setup ---

HMODULE LoadCudaDll() {
  HMODULE dll = LoadLibraryA("cudart64_12.dll");
  if (dll) return dll;
  return LoadLibraryA("cudart.dll");
}

template <typename T>
bool Resolve(HMODULE dll, const char* name, T* out) {
  *out = reinterpret_cast<T>(GetProcAddress(dll, name));
  return *out != nullptr;
}

bool InstallHook(const char* name, void* hook, void** raw_tramp,
                 void** typed_tramp) {
  void* target = GetProcAddress(g_cuda_dll, name);
  if (!target) return false;
  void* tramp = HookInstall(target, hook);
  if (!tramp) return false;
  *raw_tramp = tramp;
  *typed_tramp = tramp;
  return true;
}

}  // namespace

int InitializeIntercept() {
  HMODULE dll = LoadCudaDll();
  if (!dll) {
    std::printf("Intercept: cannot load CUDA runtime DLL\n");
    return 1;
  }
  g_cuda_dll = dll;

  bool ok = true;
  ok &= Resolve(dll, "cudaMalloc", &g_real_malloc);
  ok &= Resolve(dll, "cudaFree", &g_real_free);
  ok &= Resolve(dll, "cudaMemcpy", &g_real_memcpy);
  ok &= Resolve(dll, "cudaLaunchKernel", &g_real_launch);
  if (!ok) return 1;

  ok &= InstallHook("cudaMalloc",
      reinterpret_cast<void*>(Hook_Malloc),
      &g_raw_tramp_malloc,
      reinterpret_cast<void**>(&g_tramp_malloc));
  ok &= InstallHook("cudaFree",
      reinterpret_cast<void*>(Hook_Free),
      &g_raw_tramp_free,
      reinterpret_cast<void**>(&g_tramp_free));
  ok &= InstallHook("cudaMemcpy",
      reinterpret_cast<void*>(Hook_Memcpy),
      &g_raw_tramp_memcpy,
      reinterpret_cast<void**>(&g_tramp_memcpy));
  ok &= InstallHook("cudaLaunchKernel",
      reinterpret_cast<void*>(Hook_Launch),
      &g_raw_tramp_launch,
      reinterpret_cast<void**>(&g_tramp_launch));

  if (!ok) {
    ShutdownIntercept();
    return 1;
  }

  g_vmm_allocator.Initialize(/*device_index=*/0);
  g_vmm_active = true;

  std::printf("Intercept: 4 functions hooked, VMM active\n");
  return 0;
}

void ShutdownIntercept() {
  if (!g_cuda_dll) return;
  if (g_raw_tramp_malloc)
    HookRemove(GetProcAddress(g_cuda_dll, "cudaMalloc"),
               g_raw_tramp_malloc);
  if (g_raw_tramp_free)
    HookRemove(GetProcAddress(g_cuda_dll, "cudaFree"),
               g_raw_tramp_free);
  if (g_raw_tramp_memcpy)
    HookRemove(GetProcAddress(g_cuda_dll, "cudaMemcpy"),
               g_raw_tramp_memcpy);
  if (g_raw_tramp_launch)
    HookRemove(GetProcAddress(g_cuda_dll, "cudaLaunchKernel"),
               g_raw_tramp_launch);
}

int RunInterceptSmoke() {
  std::printf("Intercept smoke: starting...\n");

  cuInit(0);
  if (InitializeIntercept() != 0) return 1;

  // Call through hooked cudaMalloc/cudaFree.
  auto call_malloc = reinterpret_cast<MallocFn>(
      GetProcAddress(g_cuda_dll, "cudaMalloc"));
  auto call_free = reinterpret_cast<FreeFn>(
      GetProcAddress(g_cuda_dll, "cudaFree"));

  void* dev = nullptr;
  int err = call_malloc(&dev, 64 * 1024 * 1024);
  if (err != kCudaSuccess || !dev) {
    std::printf("  cudaMalloc failed: %d\n", err);
    return 1;
  }
  std::printf("  cudaMalloc(64MB) -> %p\n", dev);

  err = call_free(dev);
  if (err != kCudaSuccess) {
    std::printf("  cudaFree failed: %d\n", err);
    return 1;
  }
  std::printf("  cudaFree OK\n");

  std::printf("  Allocs: %zu  Frees: %zu  Memcpys: %zu\n",
              GetInterceptAllocCount(), GetInterceptFreeCount(),
              GetInterceptMemcpyCount());

  ShutdownIntercept();
  std::printf("Intercept smoke: PASS\n");
  return 0;
}

std::size_t GetInterceptAllocCount() { return g_alloc_count; }
std::size_t GetInterceptFreeCount() { return g_free_count; }
std::size_t GetInterceptMemcpyCount() { return g_memcpy_count; }
std::size_t GetInterceptLaunchCount() { return g_launch_count; }

}  // namespace vmm_project
