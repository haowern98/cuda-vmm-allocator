#ifndef VMM_PROJECT_BACKEND_BACKEND_RUNNER_H_
#define VMM_PROJECT_BACKEND_BACKEND_RUNNER_H_

#include <cstdio>
#include <string>

#include "util/timer.h"

namespace vmm_project {

struct BackendResult {
  bool ok = false;
  double load_ms = 0.0;
  double run_ms = 0.0;
  double unload_ms = 0.0;
};

template <typename Backend>
BackendResult RunBackend(const std::string& model_path,
                         const std::string& prompt) {
  BackendResult result;
  Backend backend;

  Timer timer;
  if (!backend.Load(model_path)) {
    std::printf("  Load: FAILED\n");
    return result;
  }
  result.load_ms = timer.ElapsedMs();
  std::printf("  Load: %.1f ms\n", result.load_ms);

  timer.Reset();
  if (!backend.RunOnce(prompt)) {
    std::printf("  RunOnce: FAILED\n");
    return result;
  }
  result.run_ms = timer.ElapsedMs();
  std::printf("  RunOnce: %.1f ms\n", result.run_ms);

  timer.Reset();
  backend.Unload();
  result.unload_ms = timer.ElapsedMs();
  std::printf("  Unload: %.1f ms\n", result.unload_ms);

  result.ok = true;
  std::printf("  Status: OK\n");
  return result;
}

}  // namespace vmm_project

#endif  // VMM_PROJECT_BACKEND_BACKEND_RUNNER_H_
