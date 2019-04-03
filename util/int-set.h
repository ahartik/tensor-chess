#ifndef UTIL_INT_SET_H
#define UTIL_INT_SET_H

#include "util/refcount.h"

#include <new>

#include <cassert>
#include <cstdlib>
#include <cstdint>

namespace int_set {

static constexpr int kKeyBits = 64;
static constexpr int kBitsPerLevel = 5;
static constexpr int kFanOut = 1 << kBitsPerLevel;

int popcount(uint64_t x) { return __builtin_popcountll(x); }

struct Node {
  mutable RefCount refcount{1};
  uint32_t set;
  Node* children[1];
};

Node* AllocNode(int size) {
  assert(size > 0);
  void* mem = malloc(sizeof(Node) + (size - 1) * sizeof(Node*));
  return new (mem) Node();
}

void FreeNode(Node* n) {
  n->~Node();
  free(reinterpret_cast<void*>(n));
}

Node* InsertCopy(Node* root, int bit_offset, uint64_t x) {
  const uint32_t bit = (x >> bit_offset) & (kFanOut - 1);
  if (root == nullptr) {
    Node* n = AllocNode(1);
    n->set = 1 << bit;
    n->children[0] = reinterpret_cast<Node*>(x);
  }
  int count = popcount(root->set);
  return nullptr;
}

}  // namespace int_set

class IntSet {
 public:
  IntSet() {}
  IntSet(const IntSet& o) : root_(o.root_) {
    if (root_ != nullptr) {
      root_->refcount.Inc();
    }
  }
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

  IntSet Insert(uint64_t x) const {
    return IntSet(InsertCopy(root_, 0, x));
  }

  bool Contains(uint64_t x) const {
    return false;
  }

 private:
  explicit IntSet(int_set::Node* r) : root_(r) {}

  int_set::Node* root_ = nullptr;
};

#endif
