#ifndef _UTIL_PERSISTENT_INT_MAP_H_
#define _UTIL_PERSISTENT_INT_MAP_H_

#include <cassert>
#include <cstdint>
#include <iostream>
#include <utility>

#include "util/refcount.h"

namespace overmind {

template <typename KeyInt, typename Value>
class PersistentIntMap {
 public:
  PersistentIntMap() : root_(nullptr) {}
  PersistentIntMap(const PersistentIntMap& other) : root_(other.root_) {
    if (root_ != nullptr) {
      root_->refcount.Inc();
    }
  }
  PersistentIntMap& operator=(const PersistentIntMap& o) {
    UnrefNode(root_);
    root_ = o.root_;
    root_->refcount.Inc();
    return *this;
  }

  ~PersistentIntMap() { UnrefNode(root_); }

  // Returns null if 'key' doesn't exist in the map.
  const Value* Find(KeyInt key) const { return DoFind(root_, key, 0); }

  PersistentIntMap Insert(KeyInt key, Value value) const {
    const NodeBase* new_root = DoInsert(root_, key, value, 0);
    return PersistentIntMap(new_root);
  }

  template <typename UpdateFunc>
  PersistentIntMap Update(KeyInt key, const UpdateFunc& func) {
    // TODO: optimize this to do only a single traversal.
    return Insert(key, func(Find(key)));
  }

  // No way to remove anything, ha ha.
 private:
  static constexpr int kKeyBits = sizeof(KeyInt) * 8;
  static constexpr int kBitsPerLevel = 4;
  static constexpr int kFanOut = 1 << kBitsPerLevel;

  struct NodeBase {
    mutable RefCount refcount{1};
    bool is_leaf = true;
  };
  struct InnerNode : public NodeBase {
    InnerNode() { NodeBase::is_leaf = false; }
    const NodeBase* children[kFanOut] = {};
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
      &existing_leaf->refcount.Inc();
      // std::cerr << "splitting leaf for " << key << " and " <<
      // existing_leaf->key << "\n";

      // Convert this to inner node, and add both existing and new leaf.
      InnerNode* new_inner = new InnerNode;
      InnerNode* last_inner = new_inner;
      while (bit_offset < kKeyBits) {
        const int digit_1 = (new_leaf->key >> bit_offset) & (kFanOut - 1);
        const int digit_2 = (existing_leaf->key >> bit_offset) & (kFanOut - 1);
        if (digit_1 != digit_2) {
          last_inner->children[digit_1] = new_leaf;
          last_inner->children[digit_2] = existing_leaf;
          return new_inner;
        } else {
          InnerNode* nl = new InnerNode;
          last_inner->children[digit_1] = nl;
          last_inner = nl;
        }
        bit_offset += kBitsPerLevel;
      }
      std::cerr << "XXX\n";
      abort();
    } else {
      // std::cerr << "Replacing inner with offset " << bit_offset << "\n";
      // 'node' is inner, create a copy with updated child.
      const InnerNode* inner = static_cast<const InnerNode*>(node);
      const int digit = (key >> bit_offset) & (kFanOut - 1);

      InnerNode* new_inner = new InnerNode;
      new_inner->children[digit] = DoInsert(inner->children[digit], key, value,
                                            bit_offset + kBitsPerLevel);
      for (int i = 0; i < kFanOut; ++i) {
        if (i != digit) {
          if (inner->children[i] != nullptr) {
            new_inner->children[i] = inner->children[i];
            inner->children[i]->refcount.Inc();
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
    if (node->refcount.Dec()) {
      if (node->is_leaf) {
        delete static_cast<const LeafNode*>(node);
      } else {
        const InnerNode* as_inner = static_cast<const InnerNode*>(node);
        for (int i = 0; i < kFanOut; ++i) {
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
    const int digit = (key >> bit_offset) & (kFanOut - 1);
    return DoFind(as_inner->children[digit], key, bit_offset + kBitsPerLevel);
  }

  const NodeBase* root_ = nullptr;
};

}  // namespace overmind

#endif  // _OVERMIND_PERSISTENT_INT_MAP_H_
