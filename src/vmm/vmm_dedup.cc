#include "vmm/vmm_dedup.h"

#include <cstring>

namespace vmm_project {

// FNV-1a 64-bit hash. Fast, good distribution, minimal code.
std::uint64_t VmmDedupTable::Hash(const void* data,
                                   std::size_t size) {
  constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
  constexpr std::uint64_t kFnvPrime  = 1099511628211ULL;
  std::uint64_t h = kFnvOffset;
  auto* p = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < size; ++i) {
    h ^= p[i];
    h *= kFnvPrime;
  }
  return h;
}

void* VmmDedupTable::Lookup(std::uint64_t hash) const {
  auto it = entries_.find(hash);
  if (it != entries_.end() && it->second.ref_count > 0) {
    return it->second.ptr;
  }
  return nullptr;
}

void VmmDedupTable::Register(std::uint64_t hash, void* ptr,
                              std::size_t bytes) {
  auto it = entries_.find(hash);
  if (it != entries_.end()) {
    it->second.ref_count += 1;
    dedup_hits_ += 1;
    bytes_saved_ += bytes;
  } else {
    Entry e;
    e.ptr = ptr;
    e.ref_count = 1;
    entries_[hash] = e;
  }
}

void VmmDedupTable::UnregisterPtr(void* ptr) {
  for (auto it = entries_.begin(); it != entries_.end(); ) {
    if (it->second.ptr == ptr) {
      it->second.ref_count -= 1;
      if (it->second.ref_count <= 0) {
        it = entries_.erase(it);
        continue;
      }
    }
    ++it;
  }
}

}  // namespace vmm_project
