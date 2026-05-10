#include "interpose/intercept_hook.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#error "intercept_hook requires Windows"
#endif

namespace vmm_project {

namespace {

// We overwrite the first 14 bytes with a JMP patch.
// The trampoline copies the first kCopyBytes bytes, executes them,
// then JMPs back to target + kCopyBytes.
// 40 bytes is enough to cover any reasonable function prologue without
// splitting instructions.
constexpr int kJmpPatchSize = 14;
constexpr int kCopyBytes = 40;
constexpr int kTrampolineSize = kCopyBytes + kJmpPatchSize;
constexpr int kPageSize = 4096;

struct TrampolinePage {
  uint8_t* base;
  std::size_t used;
  TrampolinePage* next;
};

TrampolinePage* g_pages = nullptr;

uint8_t* AllocateTrampolineBlock() {
  TrampolinePage* page = g_pages;
  if (!page || page->used + kTrampolineSize > kPageSize) {
    void* mem = VirtualAlloc(nullptr, kPageSize,
                             MEM_COMMIT | MEM_RESERVE,
                             PAGE_EXECUTE_READWRITE);
    if (!mem) {
      std::printf("Hook: VirtualAlloc failed (%lu)\n",
                  GetLastError());
      return nullptr;
    }
    auto* np = new TrampolinePage();
    np->base = static_cast<uint8_t*>(mem);
    np->used = 0;
    np->next = g_pages;
    g_pages = np;
    page = np;
  }
  uint8_t* result = page->base + page->used;
  page->used += kTrampolineSize;
  return result;
}

}  // namespace

void* HookInstall(void* target, void* hook) {
  if (!target || !hook) return nullptr;

  uint8_t* code = static_cast<uint8_t*>(target);

  // Allocate trampoline.
  uint8_t* tramp = AllocateTrampolineBlock();
  if (!tramp) return nullptr;

  // Copy kCopyBytes of original code.
  std::memcpy(tramp, code, kCopyBytes);

  // Append absolute JMP back to code + kCopyBytes.
  uint8_t* jmp = tramp + kCopyBytes;
  jmp[0] = 0xFF;
  jmp[1] = 0x25;
  jmp[2] = 0x00;
  jmp[3] = 0x00;
  jmp[4] = 0x00;
  jmp[5] = 0x00;
  uintptr_t back_addr =
      reinterpret_cast<uintptr_t>(code + kCopyBytes);
  std::memcpy(jmp + 6, &back_addr, sizeof(back_addr));

  FlushInstructionCache(GetCurrentProcess(), tramp,
                        kTrampolineSize);

  // Write JMP patch at target.
  DWORD old = 0;
  VirtualProtect(code, kJmpPatchSize, PAGE_EXECUTE_READWRITE,
                 &old);
  code[0] = 0xFF;
  code[1] = 0x25;
  code[2] = 0x00;
  code[3] = 0x00;
  code[4] = 0x00;
  code[5] = 0x00;
  uintptr_t hook_addr = reinterpret_cast<uintptr_t>(hook);
  std::memcpy(code + 6, &hook_addr, sizeof(hook_addr));
  FlushInstructionCache(GetCurrentProcess(), code,
                        kJmpPatchSize);
  DWORD unused = 0;
  VirtualProtect(code, kJmpPatchSize, old, &unused);

  return tramp;
}

void HookRemove(void* target, void* trampoline) {
  if (!target || !trampoline) return;
  uint8_t* code = static_cast<uint8_t*>(target);
  uint8_t* tramp = static_cast<uint8_t*>(trampoline);

  DWORD old = 0;
  VirtualProtect(code, kJmpPatchSize, PAGE_EXECUTE_READWRITE,
                 &old);
  std::memcpy(code, tramp, kJmpPatchSize);
  FlushInstructionCache(GetCurrentProcess(), code,
                        kJmpPatchSize);
  DWORD unused = 0;
  VirtualProtect(code, kJmpPatchSize, old, &unused);
}

}  // namespace vmm_project
