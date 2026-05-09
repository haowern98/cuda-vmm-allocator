#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include "cuda/cuda_device.h"
#include "util/timer.h"
#include "vmm/vmm_allocator.h"
#include "vmm/vmm_probe.h"

namespace {

void PrintUsage() {
  std::printf(
      "vmm-project — CUDA Virtual Memory Hypervisor for Local AI\n");
  std::printf("Usage: vmm-project.exe <command> [options]\n");
  std::printf("\nCommands:\n");
  std::printf("  probe        Query GPU for VMM support\n");
  std::printf("  alloc-smoke  Stress-test the VMM allocator\n");
  std::printf("  version      Print version\n");
  std::printf("\n");
}

void PrintVersion() {
  std::printf("vmm-project version 0.1.0\n");
}

int RunAllocSmoke(int argc, char** argv) {
  // Parse arguments: --budget <MB> --blocks <N>
  std::size_t budget_mb = 4096;
  int block_count = 1000;
  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--budget") == 0 && i + 1 < argc) {
      budget_mb = static_cast<std::size_t>(std::atol(argv[++i]));
    } else if (std::strcmp(argv[i], "--blocks") == 0 &&
               i + 1 < argc) {
      block_count = std::atoi(argv[++i]);
    }
  }

  CUresult init_result = cuInit(0);
  if (init_result != CUDA_SUCCESS) {
    std::printf("cuInit failed — no CUDA driver?\n");
    return 1;
  }

  std::size_t budget_bytes = budget_mb * 1024 * 1024;
  std::printf("Alloc-smoke: budget=%zu MiB, blocks=%d\n",
              budget_mb, block_count);

  vmm_project::VmmAllocator allocator;
  allocator.Initialize(/*device_index=*/0);

  std::mt19937_64 rng(42);
  std::uniform_int_distribution<std::size_t> size_dist(
      1024, 32 * 1024 * 1024);  // 1KB to 32MB
  std::uniform_real_distribution<double> flip(0.0, 1.0);

  struct LiveBlock {
    void* ptr;
    std::size_t size;
  };
  std::vector<LiveBlock> live;
  live.reserve(block_count);

  vmm_project::Timer timer;
  int allocs = 0;
  int frees = 0;
  int budget_rejections = 0;

  for (int i = 0; i < block_count; ++i) {
    // Occasionally free instead of allocate.
    if (!live.empty() && flip(rng) < 0.3) {
      std::uniform_int_distribution<std::size_t> idx(
          0, live.size() - 1);
      std::size_t j = idx(rng);
      allocator.Deallocate(live[j].ptr);
      live[j] = live.back();
      live.pop_back();
      ++frees;
      continue;
    }

    std::size_t size = size_dist(rng);
    if (allocator.stats().live_bytes + size > budget_bytes) {
      ++budget_rejections;
      continue;
    }

    char tag[32];
    std::snprintf(tag, sizeof(tag), "smoke-%d", i);
    void* ptr = allocator.Allocate(size, tag);
    if (ptr) {
      live.push_back({ptr, size});
      ++allocs;
    } else {
      ++budget_rejections;
    }
  }

  // Free remaining.
  for (auto& block : live) {
    allocator.Deallocate(block.ptr);
    ++frees;
  }
  live.clear();

  double elapsed_ms = timer.ElapsedMs();
  auto stats = allocator.stats();

  std::printf("  Elapsed:        %.1f ms\n", elapsed_ms);
  std::printf("  Allocations:     %d\n", allocs);
  std::printf("  Frees:           %d\n", frees);
  std::printf("  Budget reject:   %d\n", budget_rejections);
  std::printf("  VMM allocs:      %zu\n", stats.vmm_count);
  std::printf("  Fallback allocs: %zu\n", stats.fallback_count);
  std::printf("  Peak live bytes: %zu MiB\n",
              stats.peak_bytes / (1024 * 1024));
  std::printf("  Final live bytes: %zu\n", stats.live_bytes);

  if (stats.live_bytes != 0) {
    std::printf("FAIL: live_bytes != 0 after full cleanup\n");
    return 1;
  }
  std::printf("PASS: alloc-smoke completed successfully\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    PrintUsage();
    return 1;
  }

  const char* command = argv[1];

  if (std::strcmp(command, "probe") == 0) {
    return vmm_project::RunProbe();
  }
  if (std::strcmp(command, "alloc-smoke") == 0) {
    return RunAllocSmoke(argc, argv);
  }
  if (std::strcmp(command, "version") == 0) {
    PrintVersion();
    return 0;
  }

  PrintUsage();
  return 1;
}
