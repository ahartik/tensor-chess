#ifndef UTIL_INT_SET_H
#define UTIL_INT_SET_H

#include "util/refcount.h"

#include <new>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace int_set {

static constexpr int kKeyBits = 64;
static constexpr int kBitsPerLevel = 5;
static constexpr int kFanOut = 1 << kBitsPerLevel;

int popcount(uint64_t x) { return __builtin_popcountll(x); }
int first_bit(uint64_t x) {
  assert(x != 0);
  return ffsll(x) - 1;
}

struct Node {
  mutable RefCount refcount{1};
  uint32_t set;
  uint32_t leaf_set;
  Node* children[1];

  uint64_t leaf_value(int i) const {
    assert();
    return reinterpret_cast<uint64_t>(children[i]);
  }
  bool has_bit(int bit) const { return (set >> bit) & 1; }
  bool child_is_leaf(int bit) const { return (leaf_set >> bit) & 1; }
  int bit_index(int bit) const {
    assert(has_bit(bit));
    uint32_t mask = (1 << bit) - 1;
    return popcount(mask & set);
  }
};

Node* AllocNode(int size) {
  assert(size > 0);
  void* mem = malloc(sizeof(Node) + (size - 1) * sizeof(Node*));
  return new (mem) Node();
}

template <typename Func>
void IterateBits(uint64_t set, Func f) {
  int i = 0;
  while (set != 0) {
    assert(i < 64);
    const int bit = first_bit(set);
    f(i, bit);
    set ^= (1ull << bit);
    ++i;
  }
}

void Ref(Node* n) {
  assert(n != nullptr);
  assert(n->refcount.value() > 0);
  n->refcount.Inc();
}

void Unref(Node* n) {
  if (n->refcount.Dec()) {
    const int count = popcount(n->set);
    assert(count > 0);
    assert(n->refcount.count() == 0);
    if (count > 1) {
      // Unref children too.
      for (int i = 0; i < count; ++i) {
        Unref(n->children[i]);
      }
    }
    // Free memory.
    n->~Node();
    free(n);
  }
}

Node* InsertCopy(Node* root, int bit_offset, uint64_t x) {
  const uint32_t bit = (x >> bit_offset) & (kFanOut - 1);
  if (root == nullptr) {
    Node* n = AllocNode(1);
    n->set = 1 << bit;
    n->is_leaf = true;
    n->children[0] = reinterpret_cast<Node*>(x);
    return n;
  }
  const int count = popcount(root->set);
  const bool has_bit = root->has_bit(bit);
  if (root->is_leaf) {
    if (has_bit) {
      // If we happen to have this exact value already, no need to do anything.
      int index 
    }
  }
  if ((root->set >> bit) & 1) {
    // Already have an inner node for this - go further.
    if (count == 1) {
    }
  }
  return nullptr;
}

}  // namespace int_set

// This class is thread-compatible (but most operations are const).
class IntSet {
 public:
  IntSet() {}
  IntSet(const IntSet& o) : root_(o.root_) {
    if (root_ != nullptr) {
      root_->refcount.Inc();
    }
  }
  ~IntSet() { Unref(root_); }
  IntSet& operator=(const IntSet& o) {
    if (o.root_ == root_) {
      return *this;
    }
    if (root_ != nullptr) {
      root_->refcount.Dec();
    }
    root_ = o.root_;
    if (root_ != nullptr) {
      root_->refcount.Inc();
    }
    return *this;
  }

  IntSet Insert(uint64_t x) const { return IntSet(InsertCopy(root_, 0, x)); }

  bool Contains(uint64_t x) const { return false; }

 private:
  explicit IntSet(int_set::Node* r) : root_(r) {}

  int_set::Node* root_ = nullptr;
};

#endif
