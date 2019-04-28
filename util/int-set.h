#ifndef UTIL_INT_SET_H
#define UTIL_INT_SET_H

#include "util/refcount.h"

#include <algorithm>
#include <array>
#include <memory>
#include <new>
#include <vector>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace int_set {
struct Node;
}  // namespace int_set

// This class is thread-compatible (but most operations are const).
class IntSet {
 public:
  IntSet();
  template <typename It>
  IntSet(It begin, It end) : IntSet(std::vector<uint64_t>(begin, end)) {}

  ~IntSet();

  IntSet(const IntSet& o);
  IntSet& operator=(const IntSet& o);

  IntSet Insert(uint64_t x) const;
  bool Contains(uint64_t x) const;

 private:
  explicit IntSet(int_set::Node* r) : root_(r) {}
  explicit IntSet(std::vector<uint64_t> vec);

  int_set::Node* root_ = nullptr;
};

namespace linked_set {

constexpr int kNodeSize = 8;
struct Node {
  Node() {}
  mutable RefCount refcount{1};
  Node* next = nullptr;
  std::array<uint64_t, kNodeSize> vals = {};
  uint64_t summary;

  Node* Ref() {
    refcount.Inc();
    return this;
  }
  void Unref() {
    if (refcount.Dec()) {
      delete this;
    }
  }

 private:
  ~Node() {
    if (next != nullptr) {
      next->Unref();
    }
  }
};

}  // namespace linked_set

// This class is thread-compatible (but most operations are const).
class SmallIntSet {
 public:
  SmallIntSet() {}
  ~SmallIntSet() {
    if (rest_ != nullptr) {
      rest_->Unref();
    }
  }

  SmallIntSet(const SmallIntSet& o) : inlined_size_(o.inlined_size_) {
    for (int i = 0; i < inlined_size_; ++i) {
      inlined_[i] = o.inlined_[i];
    }
    rest_ = o.rest_;
    if (rest_ != nullptr) {
      rest_->Ref();
    }
  }

  SmallIntSet& operator=(const SmallIntSet& o) {
    if (&o == this) {
      return *this;
    }
    if (rest_ != nullptr) {
      rest_->Unref();
    }
    rest_ = o.rest_;
    if (rest_ != nullptr) {
      rest_->Ref();
    }
    inlined_size_ = o.inlined_size_;
    for (int i = 0; i < inlined_size_; ++i) {
      inlined_[i] = o.inlined_[i];
    }
  }

  SmallIntSet(SmallIntSet&& o)
      : inlined_(o.inlined_), inlined_size_(o.inlined_size_), rest_(o.rest_) {
    o.rest_ = nullptr;
    o.inlined_size_ = 0;
  }

  SmallIntSet& operator=(SmallIntSet&& o) {
    inlined_ = o.inlined_;
    inlined_size_ = o.inlined_size_;
    rest_ = o.rest_;

    o.rest_ = nullptr;
    o.inlined_size_ = 0;
    return *this;
  }

  SmallIntSet Insert(uint64_t x) const {
    SmallIntSet o = *this;
    if (o.inlined_size_ == static_cast<int>(inlined_.size())) {
      auto* n = new linked_set::Node;
      for (int i = 0; i < o.inlined_size_; ++i) {
        n->vals[i] = o.inlined_[i];
      }
      n->vals[o.inlined_size_] = x;
      n->summary = 0;
      for (size_t i = 0; i < n->vals.size(); ++i) {
        n->summary |= (n->vals[i] & 0xff) << (8 * i);
      }
      n->next = rest_;

      o.inlined_size_ = 0;
      o.rest_ = n;
    } else {
      o.inlined_[o.inlined_size_] = x;
      ++o.inlined_size_;
    }
    return o;
  }

  bool Contains(uint64_t x) const {
    for (int i = 0; i < inlined_size_; ++i) {
      if (x == inlined_[i]) {
        return true;
      }
    }
    auto* n = rest_;
    // TODO: Explain this.
    static constexpr uint64_t kByteOnes = (~0ULL / 255);
    const uint64_t mask = kByteOnes * (x & 0xff);
    // From https://graphics.stanford.edu/~seander/bithacks.html
    static const auto has_zero = [](uint64_t v) {
      return ((v - kByteOnes) & ~v & (kByteOnes << 7));
    };
    while (n != nullptr) {
      if (has_zero(mask ^ n->summary)) {
        for (uint64_t v : n->vals) {
          if (x == v) {
            return true;
          }
        }
      }
      n = n->next;
    }
    return false;
  }

 private:
  std::array<uint64_t, linked_set::kNodeSize - 1> inlined_;
  int inlined_size_ = 0;
  linked_set::Node* rest_ = nullptr;
};

#endif
