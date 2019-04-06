#include "c4cc/model.h"

#include <sys/stat.h>
#include <sys/types.h>

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
  tensorflow::SessionOptions opts;
  opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.2);
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
  absl::ReaderMutexLock lock(&mu_);

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

  num_preds_.fetch_add(num_boards, std::memory_order::memory_order_relaxed);
  return pred;
}

void Model::RunTrainStep(const tensorflow::Tensor& board_batch,
                         const tensorflow::Tensor& move_batch,
                         const tensorflow::Tensor& value_batch) {
  absl::MutexLock lock(&mu_);
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
      {"total_loss"}, {"train"}, &out_tensors));
  CHECK_EQ(out_tensors.size(), 1);
  const auto& total_loss = out_tensors[0];
  CHECK_EQ(total_loss.dims(), 1);
  CHECK_EQ(total_loss.dim_size(0), batch_size);
  double loss_sum = 0;
  for (int i = 0; i < batch_size; ++i) {
    loss_sum += total_loss.flat<float>()(i);
  }
  LOG(INFO) << "Training avg loss: " << (loss_sum / batch_size);
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

std::string GetCheckpointDir(int gen) {
  // TODO: Create flags out of these.
  const std::string prefix = "/mnt/tensor-data/c4cc";
  if (gen < 0) {
    return prefix + "/current";
  }
  return prefix + "/" + std::to_string(gen);
}

}  // namespace

std::string GetDefaultGraphDef() {
  // TODO: Create flags out of these.
  const std::string prefix = "/mnt/tensor-data/c4cc";
  return prefix + "/graph.pb";
}

std::string GetDefaultCheckpoint(int gen) {
  return GetCheckpointDir(gen) + "/checkpoint";
}

std::unique_ptr<Model> CreateDefaultModel(bool allow_init, int gen,
                                          const std::string& dir) {
  if (allow_init) {
    CHECK_LT(gen, 0) << "Initialization only allowed for the current gen";
  }
  const std::string prefix =
      "/mnt/tensor-data/c4cc" + (dir.empty() ? "" : "/" + dir);
  const std::string checkpoint_dir = GetCheckpointDir(gen);
  const std::string checkpoint_prefix = GetDefaultCheckpoint(gen);
  bool restore = DirectoryExists(checkpoint_dir);
  std::unique_ptr<Model> model = absl::make_unique<Model>(GetDefaultGraphDef());
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
    if (!DirectoryExists(GetCheckpointDir(g))) {
      return g;
    }
  }
}

tensorflow::Tensor MakeBoardTensor(int batch_size) {
  return tensorflow::Tensor(tensorflow::DT_FLOAT,
                            tensorflow::TensorShape({batch_size, 84}));
}

void BoardToTensor(const Board& b, tensorflow::Tensor* tensor, int i) {
  static const Color kColorOrder[2][2] = {
      {Color::kOne, Color::kTwo},
      {Color::kTwo, Color::kOne},
  };
  int j = 0;
  for (Color c : kColorOrder[b.turn() == Color::kOne]) {
    for (int x = 0; x < 7; ++x) {
      for (int y = 0; y < 6; ++y) {
        const bool set = b.color(x, y) == c;
        tensor->matrix<float>()(i, j) = set ? 1.0 : 0.0;
        ++j;
      }
    }
  }
}

void ReadPredictions(const Model::Prediction& tensor_pred,
                     Prediction* out_arr) {
  const int n = tensor_pred.move_p.dim_size(0);
  for (int i = 0; i < n; ++i) {
    for (int m = 0; m < 7; ++m) {
      out_arr[i].move_p[m] = tensor_pred.move_p.matrix<float>()(i, m);
    }
    out_arr[i].value = tensor_pred.value.flat<float>()(i);
  }
}

bool ShufflingTrainer::Train(const Board& b, const Prediction& target) {
  data_.push_back({b, target});
  if (data_.size() >= shuffle_size_) {
    TrainBatch(batch_size_);
    return true;
  } else {
    return false;
  }
}

void ShufflingTrainer::Flush() {
  while (data_.size() >= batch_size_) {
    TrainBatch(batch_size_);
  }
  if (data_.size() > 0) {
    TrainBatch(data_.size());
  }
}

void ShufflingTrainer::TrainBatch(int size) {
  tensorflow::Tensor board_tensor(tensorflow::DT_FLOAT,
                                  tensorflow::TensorShape({size, 84}));
  tensorflow::Tensor move_tensor(tensorflow::DT_FLOAT,
                                 tensorflow::TensorShape({size, 7}));
  tensorflow::Tensor value_tensor(tensorflow::DT_FLOAT,
                                  tensorflow::TensorShape({size}));
  for (int i = 0; i < size; ++i) {
    std::uniform_int_distribution<int> dist(0, data_.size() - 1);
    const int x = dist(rng_);
    BoardToTensor(data_[x].b, &board_tensor, i);
    for (int m = 0; m < 7; ++m) {
      move_tensor.matrix<float>()(i, m) = data_[x].target.move_p[m];
    }
    value_tensor.flat<float>()(i) = data_[x].target.value;

    std::swap(data_[x], data_.back());
    data_.pop_back();
  }
  model_->RunTrainStep(board_tensor, move_tensor, value_tensor);
}

}  // namespace c4cc
