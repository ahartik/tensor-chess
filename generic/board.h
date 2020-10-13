#ifndef _GENERIC_BOARD_H_
#define _GENERIC_BOARD_H_

#include <iomanip>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/numeric/int128.h"

namespace tensorflow {
class Tensor;
class TensorShape;
}  // namespace tensorflow

namespace generic {

using BoardFP = absl::uint128;

struct PredictionResult {
  std::vector<std::pair<int, float>> policy;
  double value = 0.0;
};

std::ostream& operator<<(std::ostream& out, const PredictionResult& res) {
  out << "{ ";
  for (auto& m : res.policy) {
    out << m.first << ":" << std::setprecision(3) << m.second << " ";
  }
  return out << "v:" << res.value << " }";
}

class Board {
 public:
  virtual ~Board() {}

  virtual std::vector<int> GetValidMoves() const = 0;

  virtual std::unique_ptr<Board> Move(int move) const = 0;

  virtual BoardFP fingerprint() const = 0;

  virtual bool is_over() const = 0;

  // Result of the game, if is_over() returns true. +1 means the player who's
  // turn it is wins, -1 if that player loses, 0 in case of a draw.
  virtual int result() const = 0;

  // Returns 0 or 1 depending on who's turn it is.
  virtual int turn() const = 0;

  // XXX:
  virtual void ToTensor(tensorflow::Tensor* t, int i) const = 0;

  // TODO: Document
  virtual void GetTensorShape(int batch_size,
                              tensorflow::TensorShape* out) const = 0;

  // Number move encodings. All moves returned by GetValidMoves(), are less
  // than this.
  virtual int num_possible_moves() const = 0;
};

}  // namespace generic

#endif
