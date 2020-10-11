#include "chess/model_collection.h"

namespace chess {

const char kModelPath[] = "/mnt/tensor-data/chess-models/graph.pb";

generic::ModelCollection* GetModelCollection() {
  static auto* m =
      new generic::ModelCollection("/mnt/tensor-data/chess/model");
  return m;
}

}  // namespace chess
