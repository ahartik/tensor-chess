#ifndef _C4CC_MODEL_H_
#define _C4CC_MODEL_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

namespace c4cc {

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

 private:
  void SaveOrRestore(const std::string& checkpoint_prefix,
                     const std::string& op_name);

  std::unique_ptr<tensorflow::Session> session_;
  tensorflow::Tensor true_{tensorflow::DT_BOOL, tensorflow::TensorShape({})};
  tensorflow::Tensor false_{tensorflow::DT_BOOL, tensorflow::TensorShape({})};
};  // namespace c4ccclassModel

}

#endif
