#ifndef VMM_PROJECT_INTERPOSE_INTERCEPT_HOOK_H_
#define VMM_PROJECT_INTERPOSE_INTERCEPT_HOOK_H_

namespace vmm_project {

// Install an inline hook on `target`, redirecting to `hook`.
// Returns a trampoline pointer that can be used to call the original
// function. Returns nullptr on failure.
void* HookInstall(void* target, void* hook);

// Remove a previously installed hook.
void HookRemove(void* target, void* original_trampoline);

}  // namespace vmm_project

#endif  // VMM_PROJECT_INTERPOSE_INTERCEPT_HOOK_H_
