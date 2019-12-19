#include "c4cc/model_collection.h"

namespace c4cc {

const char kModelPath[] = "/mnt/tensor-data/c4cc/graph.pb";

generic::ModelCollection* GetModelCollection() {
  static auto* m = new generic::ModelCollection("/mnt/tensor-data/c4cc/new");
  return m;
}

}  // namespace c4cc
