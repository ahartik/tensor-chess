#include "chess/model_collection.h"

namespace chess {

const char kModelPath[] = "/mnt/tensor-data/chess-models/graph.pb";

generic::ModelCollection* GetModelCollection() {
  static auto* m = new generic::ModelCollection("/mnt/tensor-data/chess/new");
  return m;
}

}  // namespace chess
