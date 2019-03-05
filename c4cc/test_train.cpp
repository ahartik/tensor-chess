#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>

#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "c4cc/board.h"
#include "c4cc/game.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "util/recordio.h"

namespace c4cc {

class Model {
 public:
  explicit Model(const std::string& graph_def_filename) {
    tensorflow::GraphDef graph_def;
    TF_CHECK_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                            graph_def_filename, &graph_def));
    session_.reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_CHECK_OK(session_->Create(graph_def));

    true_.flat<bool>()(0) = true;
    false_.flat<bool>()(0) = false;
  }

  void Init() { TF_CHECK_OK(session_->Run({}, {}, {"init"}, nullptr)); }

  void Restore(const std::string& checkpoint_prefix) {
    SaveOrRestore(checkpoint_prefix, "save/restore_all");
  }

  std::vector<int> Predict(const std::vector<float>& batch) {
    std::vector<tensorflow::Tensor> out_tensors;
    TF_CHECK_OK(session_->Run(
        {
            {"board", MakeBoardTensor(batch)},
            {"is_training", false_},
        },
        {"output_move"}, {}, &out_tensors));
    const int width = 2 * 42;
    const int size = batch.size();
    const int num_boards = size / width;
    CHECK_EQ(size % width, 0);
    CHECK_EQ(out_tensors[0].dims(), 2);
    CHECK_EQ(out_tensors[0].dim_size(0), num_boards);
    CHECK_EQ(out_tensors[0].dim_size(1), 7);
    const auto& result = out_tensors[0].matrix<float>();
    std::vector<int> output(num_boards);
    for (int i = 0; i < num_boards; ++i) {
      int best_move = 0;
      float best_prob = 0;
      float total = 0;
      for (int j = 0; j < 7; ++j) {
        const float p = result(i, j);
        total += p;
        CHECK(!isnan(p));
        if (p > best_prob) {
          best_prob = p;
          best_move = j;
        }
      }
      output[i] = best_move;
      // std::cout << "tot=" << total << " best=" << best_move << "\n";
    }
    return output;
  }

  void RunTrainStep(const std::vector<float>& board_batch,
                    const std::vector<int>& target_batch) {
    TF_CHECK_OK(session_->Run(
        {
            {"board", MakeBoardTensor(board_batch)},
            {"target_move", MakeMoveTensor(target_batch)},
            {"is_training", true_},
        },
        {}, {"train"}, nullptr));
  }

  void Checkpoint(const std::string& checkpoint_prefix) {
    SaveOrRestore(checkpoint_prefix, "save/control_dependency");
  }

 private:
  tensorflow::Tensor MakeBoardTensor(const std::vector<float>& batch) {
    const int width = 2 * 42;
    const int size = batch.size();
    CHECK_EQ(size % width, 0);
    tensorflow::Tensor t(tensorflow::DT_FLOAT,
                         tensorflow::TensorShape({size / width, width}));
    for (int i = 0; i < batch.size(); ++i) {
      t.flat<float>()(i) = batch[i];
    }
    return t;
  }
  tensorflow::Tensor MakeMoveTensor(const std::vector<int>& batch) {
    tensorflow::Tensor t(tensorflow::DT_FLOAT,
                         tensorflow::TensorShape({(int)batch.size(), 7}));
    for (int i = 0; i < batch.size() * 7; ++i) {
      t.flat<float>()(i) = 0.0;
    }
    for (int i = 0; i < batch.size(); ++i) {
      t.flat<float>()(i * 7 + batch[i]) = 1.0;
    }
    return t;
  }
  void SaveOrRestore(const std::string& checkpoint_prefix,
                     const std::string& op_name) {
    tensorflow::Tensor t(tensorflow::DT_STRING, tensorflow::TensorShape());
    t.scalar<std::string>()() = checkpoint_prefix;
    TF_CHECK_OK(session_->Run({{"save/Const", t}}, {}, {op_name}, nullptr));
  }
  std::unique_ptr<tensorflow::Session> session_;
  tensorflow::Tensor true_{tensorflow::DT_BOOL, tensorflow::TensorShape({})};
  tensorflow::Tensor false_{tensorflow::DT_BOOL, tensorflow::TensorShape({})};
};

void BoardToTensor(const Board& b, std::vector<float>* tensor) {
  const Color kColorOrder[2][2] = {
      {Color::kOne, Color::kTwo},
      {Color::kTwo, Color::kOne},
  };
  for (Color c : kColorOrder[b.turn() == Color::kOne]) {
    for (int x = 0; x < 7; ++x) {
      for (int y = 0; y < 6; ++y) {
        const bool set = b.color(x, y) == c;
        tensor->push_back(set ? 1.0 : 0.0);
      }
    }
  }
}

class ShufflingTrainer {
 public:
  static constexpr int kShuffleBatch = 10000;
  static constexpr int kTrainBatch = 256;

  explicit ShufflingTrainer(Model* model) : model_(model) {}

  void Train(const Board& b, int move) {
    auto d = std::make_unique<BoardData>();
    for (int x = 0; x < 7; ++x) {
      for (int y = 0; y < 6; ++y) {
        d->board[x * 6 + y] = b.color(x, y);
      }
    }
    d->turn = b.turn();
    d->move = move;
    boards_.push_back(std::move(d));
    if (boards_.size() >= kShuffleBatch) {
      TrainBatch();
    }
  }

  void Flush() {
    while (boards_.size() >= kTrainBatch) {
      TrainBatch();
    }
  }

 private:
  struct BoardData {
    Color board[42];
    Color turn;
    int move;
  };

  void BoardToTensor(const BoardData& b, std::vector<float>* tensor) {
    const Color kColorOrder[2][2] = {
        {Color::kOne, Color::kTwo},
        {Color::kTwo, Color::kOne},
    };
    for (Color c : kColorOrder[b.turn == Color::kOne]) {
      for (int x = 0; x < 42; ++x) {
        const bool set = b.board[x] == c;
        tensor->push_back(set ? 1.0 : 0.0);
      }
    }
  }

  void TrainBatch() {
    std::vector<float> batch_data;
    std::vector<int> moves;
    batch_data.reserve(kTrainBatch * 2 * 42);
    moves.reserve(kTrainBatch);

    for (int i = 0; i < kTrainBatch; ++i) {
      std::uniform_int_distribution<int> dist(0, boards_.size() - 1);
      const int x = dist(rng_);
      BoardToTensor(*boards_[x], &batch_data);
      moves.push_back(boards_[x]->move);

      std::swap(boards_[x], boards_.back());
      boards_.pop_back();
    }
    model_->RunTrainStep(batch_data, moves);
  }

  std::mt19937 rng_;
  std::vector<std::unique_ptr<BoardData>> boards_;
  Model* const model_;
};

bool DirectoryExists(const std::string& dir) {
  struct stat buf;
  return stat(dir.c_str(), &buf) == 0;
}

std::vector<std::string> ListDir(const std::string& dir) {
  DIR* d = opendir(dir.c_str());
  std::vector<std::string> res;

  struct dirent* ent;
  while ((ent = readdir(d)) != nullptr) {
    std::string name(ent->d_name);
    const char* prefix = "games";
    if (strncmp(name.c_str(), prefix, strlen(prefix)) == 0) {
      res.push_back(dir + "/" + std::string(ent->d_name));
    }
  }

  CHECK_EQ(closedir(d), 0);
  return res;
}

void Go() {
  const std::string prefix = "/mnt/tensor-data/c4cc";
  const std::string graph_def_filename = prefix + "/graph.pb";
  const std::string checkpoint_dir = prefix + "/checkpoints";
  const std::string checkpoint_prefix = checkpoint_dir + "/checkpoint";
  bool restore = DirectoryExists(checkpoint_dir);

  const std::string test_file = prefix + "/games/test_set.recordio";

  std::cout << "Loading graph\n";
  Model model(graph_def_filename);
  ShufflingTrainer trainer(&model);

  if (!restore) {
    std::cout << "Initializing model weights\n";
    model.Init();
  } else {
    std::cout << "Restoring model weights from checkpoint\n";
    model.Restore(checkpoint_prefix);
  }

  std::cout << "Training for a few steps\n";

  std::mt19937 rand;

  while (true) {
    {
      auto files = ListDir(prefix + "/games");
      const std::string games_file = files[rand() % files.size()];
      util::RecordReader reader(games_file);

      std::string buf;
      int game_no = 1;
      while (reader.Read(buf)) {
        if (game_no % 1000 == 0) {
          std::cout << "Game #" << game_no << "\n";
        }
        ++game_no;
        GameRecord record;
        record.ParseFromString(buf);
        Board board;
        const Color good_ai =
            record.players(0) == GameRecord::MINMAX ? Color::kOne : Color::kTwo;
        for (const int move : record.moves()) {
          CHECK(!board.is_over());
          const bool train = board.turn() == good_ai;
          if (train) {
            trainer.Train(board, move);
          }
          board.MakeMove(move);
        }
        CHECK(board.is_over());
      }
      trainer.Flush();
    }

    std::cout << "Saving checkpoint\n";
    model.Checkpoint(checkpoint_prefix);

    std::cout << "Running test\n";
    {
      util::RecordReader reader(test_file);
      int correct = 0;
      int total = 0;

      std::string buf;
      int game_no = 1;
      while (reader.Read(buf)) {
        if (game_no % 1000 == 0) {
          std::cout << "Game #" << game_no << "\n";
        }
        ++game_no;
        GameRecord record;
        record.ParseFromString(buf);
        std::vector<float> board_tensors;
        std::vector<int> expected_moves;
        Board board;
        const Color good_ai =
            record.players(0) == GameRecord::MINMAX ? Color::kOne : Color::kTwo;
        for (const int move : record.moves()) {
          CHECK(!board.is_over());
          const bool train = board.turn() == good_ai;
          if (train) {
            BoardToTensor(board, &board_tensors);
            expected_moves.push_back(move);
          }
          board.MakeMove(move);
        }
        CHECK(board.is_over());

        const auto actual_moves = model.Predict(board_tensors);
        CHECK_EQ(actual_moves.size(), expected_moves.size());
        for (int i = 0; i < actual_moves.size(); ++i) {
          if (actual_moves[i] == expected_moves[i]) {
            ++correct;
          }
          ++total;
        }
      }
      std::cout << "Accuracy: " << (100 * double(correct) / total) << "%\n";
    }
  }
}

}  // namespace c4cc

int main(int argc, char* argv[]) {
  // Setup global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  c4cc::Go();
  return 0;
}
