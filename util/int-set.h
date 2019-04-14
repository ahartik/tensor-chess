#ifndef UTIL_INT_SET_H
#define UTIL_INT_SET_H

#include "util/refcount.h"

#include <new>

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
  IntSet(const IntSet& o);
  ~IntSet();
  IntSet& operator=(const IntSet& o);

  IntSet Insert(uint64_t x) const;
  bool Contains(uint64_t x) const;

 private:
  explicit IntSet(int_set::Node* r) : root_(r) {}

  int_set::Node* root_ = nullptr;
};

#endif
