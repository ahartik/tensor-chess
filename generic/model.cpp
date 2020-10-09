#include <sys/stat.h>
#include <sys/types.h>

#include <iterator>

#include "absl/strings/str_format.h"
#include "absl/strings/strip.h"
#include "generic/model.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

namespace generic {

namespace {

bool DirectoryExists(const std::string& dir) {
  struct stat buf;
  return stat(dir.c_str(), &buf) == 0;
}

absl::string_view StripTrailingSlash(absl::string_view s) {
  return absl::StripSuffix(s, "/");
}

}  // namespace

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

// void Model::Restore(const std::string& checkpoint_prefix) {
//   SaveOrRestore(checkpoint_prefix, "save/restore_all");
// }

Model::Prediction Model::Predict(const tensorflow::Tensor& batch) {
  // absl::ReaderMutexLock lock(&mu_);
  CHECK_GE(batch.dims(), 2);
  // CHECK_EQ(batch.dim_size(2), 64);
  const int num_boards = batch.dim_size(0);
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
  // absl::MutexLock lock(&mu_);
  const int batch_size = board_batch.dim_size(0);
  CHECK_GE(batch_size, 0);
  CHECK_EQ(move_batch.dim_size(0), batch_size);
  CHECK_EQ(value_batch.dims(), 1);
  CHECK_EQ(value_batch.dim_size(0), batch_size);

  const std::vector<std::string> debug_vars = {
      "total_loss", "value_loss", "policy_loss",
      // "l2_loss",
  };

  std::vector<tensorflow::Tensor> out_tensors;
  TF_CHECK_OK(session_->Run(
      {
          {"board", board_batch},
          {"target_move", move_batch},
          {"target_value", value_batch},
          {"is_training", true_},
      },
      debug_vars, {"train"}, &out_tensors));
  for (int i = 0; i < debug_vars.size(); ++i) {
    double sum = 0;
    const auto& tensor = out_tensors[i];
    for (int j = 0; j < batch_size; ++j) {
      sum += tensor.flat<float>()(j);
    }
    LOG(INFO) << debug_vars[i] << ": " << (sum / batch_size);
  }
}

void Model::Checkpoint(const std::string& checkpoint_dir) {
  const std::string checkpoint_prefix =
      absl::StrFormat("%s/checkpoint", StripTrailingSlash(checkpoint_dir));
  SaveOrRestore(checkpoint_prefix, "save/control_dependency");
}

void Model::SaveOrRestore(const std::string& checkpoint_prefix,
                          const std::string& op_name) {
  tensorflow::Tensor t(tensorflow::DT_STRING, tensorflow::TensorShape());
  t.scalar<tensorflow::tstring>()() = checkpoint_prefix;
  TF_CHECK_OK(session_->Run({{"save/Const", t}}, {}, {op_name}, nullptr));
}


// static
std::unique_ptr<Model> Model::Open(const std::string& graph_def_file,
                                   const std::string& checkpoint_dir) {
  const std::string checkpoint_prefix =
      absl::StrFormat("%s/checkpoint", StripTrailingSlash(checkpoint_dir));

  const bool restore = DirectoryExists(checkpoint_dir);
  std::unique_ptr<Model> model(new Model(graph_def_file));
  if (!restore) {
    LOG(ERROR) << "No network data found in " << checkpoint_dir;
    return nullptr;
  } else {
    std::cout << "Restoring model weights from checkpoint\n";
    model->SaveOrRestore(checkpoint_prefix, "save/restore_all");
  }
  return model;
}

std::unique_ptr<Model> Model::New(const std::string& graph_def_file) {
  std::unique_ptr<Model> model(new Model(graph_def_file));
  model->Init();
  return model;
}

ModelCollection::ModelCollection(std::string dir)
    : base_dir_(StripTrailingSlash(dir)) {}

int ModelCollection::CountNumGens() const {
  for (int g = 0;; ++g) {
    const std::string path = absl::StrFormat("%s/%i/", base_dir_, g);
    if (!DirectoryExists(path)) {
      return g;
    }
  }
}

std::string ModelCollection::CurrentCheckpointDir() const {
  return base_dir_ + "/current";
}

std::string ModelCollection::GenCheckpointDir(int gen) const {
  return absl::StrFormat("%s/%i/", base_dir_, gen);
}

}  // namespace generic
