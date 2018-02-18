#ifndef _OVERMIND_CHESS_H_
#define _OVERMIND_CHESS_H_

#include <iostream>
#include <atomic>
#include <utility>
#include <cassert>
#include <cstdint>

namespace overmind {

using RefCount = std::atomic<uint32_t>;

// Increments reference count.
void RefCountInc(RefCount* count) {
  std::atomic_fetch_sub_explicit(count, 1u, std::memory_order_relaxed);
}

// Decrements reference count, and return true if it hit zero.
bool RefCountDec(RefCount* count) {
  // https://stackoverflow.com/questions/10268737/c11-atomics-and-intrusive-shared-pointer-reference-count
  if (std::atomic_fetch_sub_explicit(count, 1u, std::memory_order_release) ==
      1) {
    std::atomic_thread_fence(std::memory_order_acquire);
    return true;
  }
  return false;
}

template <typename KeyInt, typename Value>
class PersistentIntMap {
 public:
  PersistentIntMap() : root_(nullptr) {}
  PersistentIntMap(const PersistentIntMap& other) : root_(other.root_) {
    if (root_ != nullptr) {
      RefCountInc(&root_->refcount);
    }
  }
  PersistentIntMap& operator=(const PersistentIntMap& o) {
    UnrefNode(root_);
    root_ = o.root_;
    RefCountInc(&root_->refcount);
  }

  ~PersistentIntMap() { UnrefNode(root_); }

  // Returns null if 'key' doesn't exist in the map.
  const Value* Find(KeyInt key) const { return DoFind(root_, key, 0); }

  PersistentIntMap Insert(KeyInt key, Value value) const {
    const NodeBase* new_root = DoInsert(root_, key, value, 0);
    return PersistentIntMap(new_root);
  }

  // No way to remove anything, ha ha.
 private:
  static constexpr int kKeyBits = sizeof(KeyInt) * 8;

  struct NodeBase {
    mutable RefCount refcount{1};
    bool is_leaf = true;
  };
  struct InnerNode : public NodeBase {
    InnerNode() { NodeBase::is_leaf = false; }
    const NodeBase* children[16] = {};
  };
  struct LeafNode : public NodeBase {
    LeafNode(KeyInt key, Value value) : key(key), value(std::move(value)) {
      NodeBase::is_leaf = true;
    }
    const KeyInt key;
    const Value value;
  };

  PersistentIntMap(const NodeBase* root) : root_(root) {}

  static const NodeBase* DoInsert(const NodeBase* node, KeyInt key,
                                  Value& value, int bit_offset) {
    if (node == nullptr) {
      return new LeafNode(key, std::move(value));
    }

    if (node->is_leaf) {
      const LeafNode* existing_leaf = static_cast<const LeafNode*>(node);
      LeafNode* new_leaf = new LeafNode(key, std::move(value));
      if (key == existing_leaf->key) {
        // std::cerr << "Matching key " << key << "\n";
        return new_leaf;
      }
      RefCountInc(&existing_leaf->refcount);
      // std::cerr << "splitting leaf for " << key << " and " << existing_leaf->key << "\n";

      // Convert this to inner node, and add both existing and new leaf.
      InnerNode* new_inner = new InnerNode;
      InnerNode* last_inner = new_inner;
      while (bit_offset < kKeyBits) {
        const int digit_1 = (new_leaf->key >> bit_offset) & 15;
        const int digit_2 = (existing_leaf->key >> bit_offset) & 15;
        if (digit_1 != digit_2) {
          last_inner->children[digit_1] = new_leaf;
          last_inner->children[digit_2] = existing_leaf;
          return new_inner;
        } else {
          InnerNode* nl = new InnerNode;
          last_inner->children[digit_1] = nl;
          last_inner = nl;
        }
        bit_offset += 4;
      }
      assert(false);
    } else {
      // std::cerr << "Replacing inner with offset " << bit_offset << "\n";
      // 'node' is inner, create a copy with updated child.
      const InnerNode* inner = static_cast<const InnerNode*>(node);
      const int digit = (key >> bit_offset) & 15;

      InnerNode* new_inner = new InnerNode;
      new_inner->children[digit] =
          DoInsert(inner->children[digit], key, value, bit_offset + 4);
      for (int i = 0; i < 16; ++i) {
        if (i != digit) {
          if (inner->children[i] != nullptr) {
            new_inner->children[i] = inner->children[i];
            RefCountInc(&inner->children[i]->refcount);
          }
        }
      }
      return new_inner;
    }
  }
  static void UnrefNode(const NodeBase* node) {
    if (node == nullptr) {
      return;
    }
    if (RefCountDec(&node->refcount)) {
      if (node->is_leaf) {
        delete static_cast<const LeafNode*>(node);
      } else {
        const InnerNode* as_inner = static_cast<const InnerNode*>(node);
        for (int i = 0; i < 16; ++i) {
          UnrefNode(as_inner->children[i]);
        }
        delete as_inner;
      }
    }
  }
  static const Value* DoFind(const NodeBase* node, KeyInt key, int bit_offset) {
    if (node == nullptr) {
      return nullptr;
    }
    if (node->is_leaf) {
      const LeafNode* as_leaf = static_cast<const LeafNode*>(node);
      if (key == as_leaf->key) {
        return &as_leaf->value;
      } else {
        return nullptr;
      }
    }
    const InnerNode* as_inner = static_cast<const InnerNode*>(node);
    const int digit = (key >> bit_offset) & 15;
    return DoFind(as_inner->children[digit], key, bit_offset + 4);
  }

  const NodeBase* root_ = nullptr;
};

}  // namespace overmind

#endif  // _OVERMIND_CHESS_H_
