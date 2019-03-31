
#include <functional>

#include "c4cc/board.h"
#include "c4cc/model.h"

namespace c4cc {

struct Prediction {
  float move_p[7];
  float value;
};

std::vector<Prediction> MakePredictions(Model* model,
                                        std::vector<const Board*> boards);

}  // namespace c4cc
