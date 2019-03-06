#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "c4cc/board.h"
#include "c4cc/game.pb.h"
#include "c4cc/model.h"
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

void BoardToTensor(const Board& b, tensorflow::Tensor* tensor, int i) {
  const Color kColorOrder[2][2] = {
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

class ShufflingTrainer {
 public:
  static constexpr int kShuffleBatch = 10000;
  static constexpr int kTrainBatch = 256;

  explicit ShufflingTrainer(Model* model) : model_(model) {}

  void Train(const Board& b, int move, Color winner) {
    auto d = std::make_unique<BoardData>();
    for (int x = 0; x < 7; ++x) {
      for (int y = 0; y < 6; ++y) {
        d->board[x * 6 + y] = b.color(x, y);
      }
    }
    d->turn = b.turn();
    d->move = move;
    if (winner == Color::kEmpty) {
      d->score = 0;
    } else {
      d->score = b.turn() == winner ? 1.0 : -1.0;
    }
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
    float score = 0.0;
  };

  void BoardToTensor(const BoardData& b, tensorflow::Tensor* tensor, int i) {
    const Color kColorOrder[2][2] = {
        {Color::kOne, Color::kTwo},
        {Color::kTwo, Color::kOne},
    };
    int j = 0;
    for (Color c : kColorOrder[b.turn == Color::kOne]) {
      for (int x = 0; x < 42; ++x) {
        const bool set = b.board[x] == c;
        tensor->matrix<float>()(i, j) = set ? 1.0 : 0.0;
        ++j;
      }
    }
  }

  void TrainBatch() {
    tensorflow::Tensor board_tensor(tensorflow::DT_FLOAT,
                                    tensorflow::TensorShape({kTrainBatch, 84}));
    tensorflow::Tensor move_tensor(tensorflow::DT_FLOAT,
                                   tensorflow::TensorShape({kTrainBatch, 7}));
    tensorflow::Tensor value_tensor(tensorflow::DT_FLOAT,
                                    tensorflow::TensorShape({kTrainBatch}));
    for (int i = 0; i < kTrainBatch; ++i) {
      std::uniform_int_distribution<int> dist(0, boards_.size() - 1);
      const int x = dist(rng_);
      BoardToTensor(*boards_[x], &board_tensor, i);
      for (int m = 0; m < 7; ++m) {
        move_tensor.matrix<float>()(i, m) = (m == boards_[x]->move) ? 1.0 : 0.0;
      }
      value_tensor.flat<float>()(i) = boards_[x]->score;

      std::swap(boards_[x], boards_.back());
      boards_.pop_back();
    }
    model_->RunTrainStep(board_tensor, move_tensor, value_tensor);
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
        const Color winner =
            record.game_result() == 0
                ? Color::kEmpty
                : record.game_result() == 1 ? Color::kOne : Color::kTwo;
        const Color good_ai =
            record.players(0) == GameRecord::MINMAX ? Color::kOne : Color::kTwo;
        for (const int move : record.moves()) {
          CHECK(!board.is_over());
          const bool train = board.turn() == good_ai;
          if (train) {
            trainer.Train(board, move, winner);
          }
          board.MakeMove(move);
        }
        CHECK(board.is_over());
      }
      trainer.Flush();
    }

    std::cout << "Saving checkpoint\n";
    model.Checkpoint(checkpoint_prefix);

#if 1
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
        tensorflow::Tensor board_tensor(tensorflow::DT_FLOAT,
                                        tensorflow::TensorShape({record.moves_size(), 84}));
        std::vector<int> expected_moves;
        Board board;
        const Color good_ai =
            record.players(0) == GameRecord::MINMAX ? Color::kOne : Color::kTwo;
        int move_no = 0;
        for (const int move : record.moves()) {
          CHECK(!board.is_over());
          const bool train = board.turn() == good_ai;
          if (train) {
            BoardToTensor(board, &board_tensor, move_no);
            ++move_no;
            expected_moves.push_back(move);
          }
          board.MakeMove(move);
        }
        CHECK(board.is_over());

        const Model::Prediction prediction  = model.Predict(board_tensor);
        std::vector<int> actual_moves;
        for (int i = 0; i < expected_moves.size(); ++i) {
          int best = 0;
          float score = 0;
          for (int m = 0; m < 7; ++m) {
            double p = prediction.move_p.matrix<float>()(i, m);
            if (p > score) {
              score = p;
              best = m;
            }
          }
          actual_moves.push_back(best);
        }
        CHECK_EQ(actual_moves.size(), expected_moves.size());
        for (int i = 0; i < actual_moves.size(); ++i) {
          if (actual_moves[i] == expected_moves[i]) {
            ++correct;
          }
          ++total;
        }
      }
      std::cout << "Accuracy: " << (100 * double(correct) / total) << "% \n";
    }
#endif
  }
}

}  // namespace c4cc

int main(int argc, char* argv[]) {
  // Setup global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  c4cc::Go();
  return 0;
}
