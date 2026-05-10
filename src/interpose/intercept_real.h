#ifndef VMM_PROJECT_INTERPOSE_INTERCEPT_REAL_H_
#define VMM_PROJECT_INTERPOSE_INTERCEPT_REAL_H_

#include <cstddef>

namespace vmm_project {

// Initialize interception. Returns 0 on success.
int InitializeIntercept();

// Remove hooks and release resources.
void ShutdownIntercept();

// Run a self-test: allocate/free/memcpy through the hooks, verify
// the counters and VMM allocator are working. Returns 0 on success.
int RunInterceptSmoke();

// Counter accessors.
std::size_t GetInterceptAllocCount();
std::size_t GetInterceptFreeCount();
std::size_t GetInterceptMemcpyCount();
std::size_t GetInterceptLaunchCount();

}  // namespace vmm_project

#endif  // VMM_PROJECT_INTERPOSE_INTERCEPT_REAL_H_
