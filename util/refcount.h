#ifndef _UTIL_REFCOUNT_H_
#define _UTIL_REFCOUNT_H_

#include <atomic>
#include <cstdint>

// This class is thread-safe.
class RefCount {
 public:
  explicit RefCount(int initial) : count_(initial) {}
  RefCount(const RefCount&) = delete;
  RefCount& operator=(const RefCount&) = delete;

  void Inc() {
    std::atomic_fetch_add_explicit(&count_, 1u, std::memory_order_relaxed);
  }
  bool Dec() {
    // https://stackoverflow.com/questions/10268737/c11-atomics-and-intrusive-shared-pointer-reference-count
    if (std::atomic_fetch_sub_explicit(&count_, 1u,
                                       std::memory_order_release) == 1) {
      std::atomic_thread_fence(std::memory_order_acquire);
      return true;
    }
    return false;
  }

  // Should only be used for debugging, not for logic.
  int count() const { return count_.load(std::memory_order_relaxed); }

 private:
  std::atomic<uint32_t> count_;
};

#endif
