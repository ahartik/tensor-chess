#ifndef _GENERIC_MODEL_H_
#define _GENERIC_MODEL_H_

#include <atomic>
#include <string>

#include "absl/synchronization/mutex.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

namespace generic {

class Model {
 public:
  struct Prediction {
    tensorflow::Tensor move_p;
    tensorflow::Tensor value;
  };

  Prediction Predict(const tensorflow::Tensor& batch);

  void RunTrainStep(const tensorflow::Tensor& board_batch,
                    const tensorflow::Tensor& move_batch,
                    const tensorflow::Tensor& value_batch);

  void Checkpoint(const std::string& dir);

  int64_t num_predictions() const {
    return num_preds_.load(std::memory_order::memory_order_relaxed);
  }

  // Opens given directory 
  static std::unique_ptr<Model> Open(const std::string& graph_def_file,
                                     const std::string& checkpoint_dir);

  // Creates a new model, initialized
  static std::unique_ptr<Model> New(const std::string& graph_def_file);

 private:
  // Creates a new model, initialized
  static std::unique_ptr<Model> OpenInternal(
      const std::string& graph_def_filename);

  explicit Model(const std::string& graph_def_filename);

  void SaveOrRestore(const std::string& checkpoint_prefix,
                     const std::string& op_name);

  void Init();

  absl::Mutex mu_;
  std::unique_ptr<tensorflow::Session> session_;
  tensorflow::Tensor true_{tensorflow::DT_BOOL, tensorflow::TensorShape({})};
  tensorflow::Tensor false_{tensorflow::DT_BOOL, tensorflow::TensorShape({})};

  std::atomic<int64_t> num_preds_{0};
};

class ModelCollection {
 public:
  explicit ModelCollection(std::string dir);

  // XXX
  int CountNumGens() const;

  std::string CurrentCheckpointDir() const;

  std::string GenCheckpointDir(int gen) const;

 private:
  const std::string base_dir_;
};

}  // namespace chess

#endif
