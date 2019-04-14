#include "util/int-set.h"

#include <algorithm>
#include <cstring>
#include <iostream>

namespace int_set {

#ifdef NDEBUG
constexpr bool kDebugMode = true;
#else
constexpr bool kDebugMode = true;
#endif

// static constexpr int kKeyBits = 64;
static constexpr int kBitsPerLevel = 5;
static constexpr int kFanOut = 1 << kBitsPerLevel;

bool IsTerminalOffset(int bit_offset) {
  return bit_offset + kBitsPerLevel >= 64;
}

inline int popcount(uint64_t x) { return __builtin_popcountll(x); }
inline int first_bit(uint64_t x) {
  assert(x != 0);
  return ffsll(x) - 1;
}

struct Node {
  mutable RefCount refcount{1};
  uint32_t set;
  uint32_t leaf_set;

  Node* child(int i) const { return reinterpret_cast<Node*>(leaf_value(i)); }

  void set_child(int i, Node* n) {
    set_leaf_value(i, reinterpret_cast<uint64_t>(n));
  }

  uint64_t leaf_value(int i) const {
    assert(i < popcount(set));
    const unsigned char* t_mem = reinterpret_cast<const unsigned char*>(this);
    uint64_t x;
    memcpy(&x, t_mem + sizeof(Node) + i * sizeof(uint64_t), sizeof(uint64_t));
    return x;
  }
  void set_leaf_value(int i, uint64_t value) {
    unsigned char* t_mem = reinterpret_cast<unsigned char*>(this);
    memcpy(t_mem + sizeof(Node) + i * sizeof(uint64_t), &value, sizeof(value));
  }

  bool has_bit(int bit) const { return (set >> bit) & 1; }
  bool child_is_leaf(int bit) const { return (leaf_set >> bit) & 1; }
  int bit_index(int bit) const {
    assert(has_bit(bit));
    uint32_t mask = (1 << bit) - 1;
    return popcount(mask & set);
  }
};

inline Node* AllocNode(int size) {
  assert(size >= 0);
  const int node_bytes = sizeof(Node) + size * sizeof(Node*);
  void* mem = malloc(node_bytes);
  memset(mem, 0xaa, node_bytes);
  return new (mem) Node();
}

template <typename Func>
void IterateBits(uint64_t set, Func f) {
  int i = 0;
  while (set != 0) {
    assert(i < 64);
    const uint32_t bit = first_bit(set);
    f(i, bit);
    set ^= (1ull << bit);
    ++i;
  }
}

Node* Ref(Node* n) {
  if (n == nullptr) {
    return nullptr;
  }
  assert(n->refcount.count() > 0);
  n->refcount.Inc();
  return n;
}

void Unref(Node* n) {
  if (n == nullptr) {
    return;
  }
  if (n->refcount.Dec()) {
    // Unref children too.
    IterateBits(n->set, [&](int i, int b) {
      if (!n->child_is_leaf(b)) {
        Unref(n->child(i));
      }
    });
    // Free memory.
    n->~Node();
    free(n);
  }
}

int GetBits(uint64_t x, int bit_offset) {
  x >>= 64 - bit_offset - kBitsPerLevel;
  return x & (kFanOut - 1);
}

Node* CreateSetFromArr(uint64_t* values, const int n, const int bit_offset) {
  if (kDebugMode) {
    const bool sorted = std::is_sorted(
        values, values + n, [bit_offset](uint64_t a, uint64_t b) {
          return GetBits(a, bit_offset) < GetBits(b, bit_offset);
        });
    if (!sorted) {
      std::cerr << "Not sorted: ";
      for (int i = 0; i < n; ++i) {
        std::cerr << values[i] << " ";
      }
      std::cerr << "\n";
      abort();
    }
  }

  if (IsTerminalOffset(bit_offset)) {
    Node* node = AllocNode(n);
    for (int i = 0; i < n; ++i) {
      const uint64_t x = values[i];
      const int bit = GetBits(x, bit_offset);
      node->leaf_set |= (1 << bit);
    }
    node->set = node->leaf_set;
    return node;
  } else {
    uint32_t set = 0;
    for (int i = 0; i < n; ++i) {
      const uint64_t x = values[i];
      const int bit = GetBits(x, bit_offset);
      set |= (1ul << bit);
    }
    int count = popcount(set);
    Node* node = AllocNode(count);
    node->set = set;
    node->leaf_set = 0;
    int start = 0;
    IterateBits(node->set, [&](int i, int b) {
      int end = start;
      for (; end < n; ++end) {
        if (GetBits(values[end], bit_offset) != b) {
          break;
        }
      }
      // values[i] for start <= i < end should go to this child.
      assert(start != end);
      if (start + 1 == end) {
        // Leaf child.
        node->leaf_set |= 1 << b;
        node->set_leaf_value(i, values[start]);
      } else {
        // Multiple children, must create an inner node.
        node->set_child(i, CreateSetFromArr(values + start, end - start,
                                            bit_offset + kBitsPerLevel));
      }
      start = end;
    });
    return node;
  }
}

Node* InsertCopy(Node* root, int bit_offset, uint64_t x) {
  const uint32_t bit = GetBits(x, bit_offset);
  if (root == nullptr) {
    Node* n = AllocNode(1);
    n->set = 1 << bit;
    n->leaf_set = 1 << bit;
    n->set_leaf_value(0, x);
    return n;
  }
  const int count = popcount(root->set);
  if (root->has_bit(bit)) {
    // If 'x' already exists as a leaf value, just return root.
    if (root->child_is_leaf(bit)) {
      const int index = popcount(root->set & ((1 << bit) - 1));
      if (root->leaf_value(index) == x) {
        return Ref(root);
      }
    }
    // Just replace that particular child.
    Node* n = AllocNode(count);
    n->leaf_set = root->leaf_set;
    n->set = root->set;
    IterateBits(n->set, [&](int i, uint32_t b) {
      if (b == bit) {
        if (!n->child_is_leaf(b)) {
          n->set_child(
              i, InsertCopy(root->child(i), bit_offset + kBitsPerLevel, x));
        } else {
          // This is a leaf child, so we need to create a new inner node.
          uint64_t values[2] = {x, root->leaf_value(i)};
          if (values[0] > values[1]) {
            std::swap(values[0], values[1]);
          }
          n->set_child(i,
                       CreateSetFromArr(values, 2, bit_offset + kBitsPerLevel));
          n->leaf_set &= ~(1 << b);
        }
      } else {
        if (n->child_is_leaf(b)) {
          n->set_leaf_value(i, root->leaf_value(i));
        } else {
          n->set_child(i, Ref(root->child(i)));
        }
      }
    });
    return n;
  } else {
    const int new_count = count + 1;
    // Add this value as a leaf.
    Node* n = AllocNode(new_count);
    n->set = root->set | (1 << bit);
    n->leaf_set = root->leaf_set | (1 << bit);

    IterateBits(n->set, [&](int i, uint32_t b) {
      if (b == bit) {
        // New value goes here.
        n->set_leaf_value(i, x);
      } else {
        const int oi = i - (b > bit ? 1 : 0);
        if (n->child_is_leaf(b)) {
          n->set_leaf_value(i, root->leaf_value(oi));
        } else {
          n->set_child(i, Ref(root->child(oi)));
        }
      }
    });
    return n;
  }
}

}  // namespace int_set

IntSet::IntSet() {}

IntSet::IntSet(const IntSet& o) : root_(Ref(o.root_)) {}

IntSet::~IntSet() { Unref(root_); }

IntSet& IntSet::operator=(const IntSet& o) {
  if (o.root_ == root_) {
    return *this;
  }
  Unref(root_);
  root_ = Ref(o.root_);
  return *this;
}

IntSet IntSet::Insert(uint64_t x) const {
  return IntSet(InsertCopy(root_, 0, x));
}

bool IntSet::Contains(uint64_t x) const { return false; }
