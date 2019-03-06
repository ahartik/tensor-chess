#include "c4cc/model.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

namespace c4cc {

Model::Model(const std::string& graph_def_filename) {
  tensorflow::GraphDef graph_def;
  TF_CHECK_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                          graph_def_filename, &graph_def));
  session_.reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_CHECK_OK(session_->Create(graph_def));

  true_.flat<bool>()(0) = true;
  false_.flat<bool>()(0) = false;
}

void Model::Init() { TF_CHECK_OK(session_->Run({}, {}, {"init"}, nullptr)); }

void Model::Restore(const std::string& checkpoint_prefix) {
  SaveOrRestore(checkpoint_prefix, "save/restore_all");
}

Model::Prediction Model::Predict(const tensorflow::Tensor& batch) {
  CHECK_EQ(batch.dims(), 2);
  const int num_boards = batch.dim_size(0);
  CHECK_EQ(batch.dim_size(1), 84);
  std::vector<tensorflow::Tensor> out_tensors;
  TF_CHECK_OK(session_->Run(
      {
          {"board", batch},
          {"is_training", false_},
      },
      {"output_move", "output_value"}, {}, &out_tensors));
  CHECK_EQ(out_tensors.size(), 2);
  Prediction pred;
  pred.move_p = out_tensors[0];
  pred.value = out_tensors[1];
  CHECK_EQ(pred.move_p.dim_size(0), num_boards);
  CHECK_EQ(pred.value.dim_size(0), num_boards);
  return pred;
}

void Model::RunTrainStep(const tensorflow::Tensor& board_batch,
                         const tensorflow::Tensor& move_batch,
                         const tensorflow::Tensor& value_batch) {
  TF_CHECK_OK(session_->Run(
      {
          {"board", board_batch},
          {"target_move", move_batch},
          {"target_value", value_batch},
          {"is_training", true_},
      },
      {}, {"train"}, nullptr));
}


void Model::Checkpoint(const std::string& checkpoint_prefix) {
  SaveOrRestore(checkpoint_prefix, "save/control_dependency");
}

void Model::SaveOrRestore(const std::string& checkpoint_prefix,
                   const std::string& op_name) {
  tensorflow::Tensor t(tensorflow::DT_STRING, tensorflow::TensorShape());
  t.scalar<std::string>()() = checkpoint_prefix;
  TF_CHECK_OK(session_->Run({{"save/Const", t}}, {}, {op_name}, nullptr));
}

}  // namespace c4cc
