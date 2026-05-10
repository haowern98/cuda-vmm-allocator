#ifndef VMM_PROJECT_VMM_VMM_DEDUP_H_
#define VMM_PROJECT_VMM_VMM_DEDUP_H_

#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace vmm_project {

class VmmDedupTable {
 public:
  VmmDedupTable() = default;

  // Compute a 64-bit hash of the given data. Fast path for dedup.
  static std::uint64_t Hash(const void* data, std::size_t size);

  // Look up a hash. Returns the VA that has this content, or null.
  void* Lookup(std::uint64_t hash) const;

  // Register a hash as having live physical storage at the given VA.
  void Register(std::uint64_t hash, void* ptr, std::size_t bytes);

  // Remove the mapping for a specific ptr (called when ptr is freed).
  void UnregisterPtr(void* ptr);

  // Statistics.
  std::size_t dedup_hits() const { return dedup_hits_; }
  std::size_t dedup_bytes_saved() const { return bytes_saved_; }
  std::size_t unique_entries() const { return entries_.size(); }

 private:
  struct Entry {
    void* ptr = nullptr;
    int ref_count = 0;
  };
  std::unordered_map<std::uint64_t, Entry> entries_;
  std::size_t dedup_hits_ = 0;
  std::size_t bytes_saved_ = 0;
};

}  // namespace vmm_project

#endif  // VMM_PROJECT_VMM_VMM_DEDUP_H_
