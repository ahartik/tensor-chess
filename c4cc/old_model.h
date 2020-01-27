#ifndef _C4CC_MODEL_H_
#define _C4CC_MODEL_H_

#include <atomic>

#include "absl/synchronization/mutex.h"
#include "c4cc/board.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

namespace c4cc {

// This class is thread-safe.
class Model {
 public:
  explicit Model(const std::string& graph_def_filename);

  void Init();

  void Restore(const std::string& checkpoint_prefix);

  struct Prediction {
    tensorflow::Tensor move_p;
    tensorflow::Tensor value;
  };

  Prediction Predict(const tensorflow::Tensor& batch);

  void RunTrainStep(const tensorflow::Tensor& board_batch,
                    const tensorflow::Tensor& move_batch,
                    const tensorflow::Tensor& value_batch);

  void Checkpoint(const std::string& checkpoint_prefix);

  int64_t num_predictions() const {
    return num_preds_.load(std::memory_order::memory_order_relaxed);
  }

 private:
  void SaveOrRestore(const std::string& checkpoint_prefix,
                     const std::string& op_name);

  absl::Mutex mu_;
  std::unique_ptr<tensorflow::Session> session_;
  tensorflow::Tensor true_{tensorflow::DT_BOOL, tensorflow::TensorShape({})};
  tensorflow::Tensor false_{tensorflow::DT_BOOL, tensorflow::TensorShape({})};

  std::atomic<int64_t> num_preds_{0};
};

tensorflow::Tensor MakeBoardTensor(int batch_size);

// Writes 'b' to tensor batch at index i. 'tensor' should have been created
// using MakeBoardTensor().
void BoardToTensor(const Board& b, tensorflow::Tensor* tensor, int i);

void ReadPredictions(const Model::Prediction& tensor_pred, Prediction* out_arr);
void ReadPredictions(const Model::Prediction& tensor_pred, Prediction* out_arr,
                     int offset, int n);

std::string GetDefaultGraphDef();
std::string GetDefaultCheckpoint(int gen = -1);
int GetNumGens();

std::unique_ptr<Model> CreateDefaultModel(bool allow_init, int gen = -1,
                                          const std::string& dir = "");

}  // namespace c4cc

#endif
