#include "chess/model.h"

#include <sys/stat.h>
#include <sys/types.h>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

namespace chess {

Model::Model(const std::string& graph_def_filename) {
  tensorflow::GraphDef graph_def;
  TF_CHECK_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                          graph_def_filename, &graph_def));
  tensorflow::SessionOptions opts;
  opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.4);
  session_.reset(tensorflow::NewSession(opts));
  TF_CHECK_OK(session_->Create(graph_def));

  true_.flat<bool>()(0) = true;
  false_.flat<bool>()(0) = false;
}

void Model::Init() { TF_CHECK_OK(session_->Run({}, {}, {"init"}, nullptr)); }

void Model::Restore(const std::string& checkpoint_prefix) {
  SaveOrRestore(checkpoint_prefix, "save/restore_all");
}

Model::Prediction Model::Predict(const tensorflow::Tensor& batch) {
  // absl::ReaderMutexLock lock(&mu_);
  CHECK_EQ(batch.dims(), 3);
  CHECK_EQ(batch.dim_size(2), 64);
  const int num_boards = batch.dim_size(0);
  std::vector<tensorflow::Tensor> out_tensors;
  TF_CHECK_OK(session_->Run(
      {
          {"board", batch},
          {"is_training", false_},
      },
      {"output_move", "output_value"},
      {}, &out_tensors));
  CHECK_EQ(out_tensors.size(), 2);
  Prediction pred;
  pred.move_p = out_tensors[0];
  pred.value = out_tensors[1];
  CHECK_EQ(pred.move_p.dim_size(0), num_boards);
  CHECK_EQ(pred.value.dim_size(0), num_boards);

  num_preds_.fetch_add(num_boards, std::memory_order::memory_order_relaxed);
  return pred;
}

void Model::RunTrainStep(const tensorflow::Tensor& board_batch,
                         const tensorflow::Tensor& move_batch,
                         const tensorflow::Tensor& value_batch) {
  // absl::MutexLock lock(&mu_);
  const int batch_size = board_batch.dim_size(0);
  CHECK_GE(batch_size, 0);
  CHECK_EQ(move_batch.dim_size(0), batch_size);
  CHECK_EQ(value_batch.dim_size(0), batch_size);

  std::vector<tensorflow::Tensor> out_tensors;
  TF_CHECK_OK(session_->Run(
      {
          {"board", board_batch},
          {"target_move", move_batch},
          {"target_value", value_batch},
          {"is_training", true_},
      },
      {"total_loss", "value_loss", "output_value"}, {"train"}, &out_tensors));
  CHECK_EQ(out_tensors.size(), 3);
  const auto& total_loss = out_tensors[0];
  const auto& value_loss = out_tensors[1];
  const auto& value = out_tensors[2];
  CHECK_EQ(total_loss.dims(), 1);
  CHECK_EQ(total_loss.dim_size(0), batch_size);
  CHECK_EQ(value_loss.dim_size(0), batch_size);
  double loss_sum = 0;
  double value_loss_sum = 0;
  double avg_value[3] = {};
  double avg_target_value[3] = {};
  for (int i = 0; i < batch_size; ++i) {
    loss_sum += total_loss.flat<float>()(i);
    value_loss_sum += value_loss.flat<float>()(i);
    for (int j = 0; j < 3; ++j) {
      avg_value[j] += value.matrix<float>()(i, j) / batch_size;
      avg_target_value[j] += value_batch.matrix<float>()(i, j) / batch_size;
    }
  }
  LOG(INFO) << "Training avg loss: " << (loss_sum / batch_size) << " value "
            << (value_loss_sum / batch_size);
  LOG(INFO) << "avg value [ "  //
            << avg_value[0] << ", " << avg_value[1] << ", " << avg_value[2]
            << " ]";
  LOG(INFO) << "avg target_value [ "  //
            << avg_target_value[0] << ", " << avg_target_value[1] << ", "
            << avg_target_value[2] << " ]";
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

namespace {
bool DirectoryExists(const std::string& dir) {
  struct stat buf;
  return stat(dir.c_str(), &buf) == 0;
}

std::string GetCheckpointDir(int gen, const std::string& dir) {
  // TODO: Create flags out of these.
  const std::string prefix = "/mnt/tensor-data/chess-models/" + dir;
  if (gen < 0) {
    return prefix + "/current";
  }
  return prefix + "/" + std::to_string(gen);
}

std::string GetGraphDef(const std::string& model_name) {
  // TODO: Create flags out of these.
  const std::string prefix = "/mnt/tensor-data/chess-models/" + model_name;
  return prefix + "/graph.pb";
}

}  // namespace

std::string GetDefaultCheckpoint(int gen, const std::string& dir) {
  return GetCheckpointDir(gen, dir) + "/checkpoint";
}


std::unique_ptr<Model> CreateDefaultModel(bool allow_init, int gen,
                                          const std::string& dir) {
  if (allow_init) {
    CHECK_LT(gen, 0) << "Initialization only allowed for the current gen";
  }
  const std::string model_name = dir.empty() ? "default" : dir;
  const std::string checkpoint_dir = GetCheckpointDir(gen, model_name);
  const std::string checkpoint_prefix = GetDefaultCheckpoint(gen, model_name);
  bool restore = DirectoryExists(checkpoint_dir);
  std::unique_ptr<Model> model =
      absl::make_unique<Model>(GetGraphDef(model_name));
  if (!restore && !allow_init) {
    LOG(FATAL)
        << "No network data found and initializing from scratch not allowed\n";
  }
  if (!restore) {
    model->Init();
  } else {
    std::cout << "Restoring model weights from checkpoint\n";
    model->Restore(checkpoint_prefix);
  }
  return model;
}

int GetNumGens() {
  for (int g = 0;; ++g) {
    if (!DirectoryExists(GetCheckpointDir(g, "default"))) {
      return g;
    }
  }
}

}  // namespace chess
