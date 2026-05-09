#ifndef VMM_PROJECT_UTIL_TIMER_H_
#define VMM_PROJECT_UTIL_TIMER_H_

#include <chrono>

namespace vmm_project {

class Timer {
 public:
  Timer() : start_(std::chrono::steady_clock::now()) {}

  void Reset() { start_ = std::chrono::steady_clock::now(); }

  double ElapsedMs() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now - start_)
        .count();
  }

 private:
  std::chrono::steady_clock::time_point start_;
};

}  // namespace vmm_project

#endif  // VMM_PROJECT_UTIL_TIMER_H_
