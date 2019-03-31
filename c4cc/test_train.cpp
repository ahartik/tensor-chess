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
#if 1
    std::cout << "Running test\n";
    {
      util::RecordReader reader(test_file);
      int correct = 0;
      int total = 0;
      double total_value_loss = 0;

      std::string buf;
      int game_no = 0;
      while (reader.Read(buf)) {
        if (game_no % 1000 == 0) {
          std::cout << "Game #" << game_no << "\n";
        }
        ++game_no;
        GameRecord record;
        record.ParseFromString(buf);
        tensorflow::Tensor board_tensor(
            tensorflow::DT_FLOAT,
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

        const Model::Prediction prediction = model.Predict(board_tensor);
        std::vector<int> actual_moves;
        std::vector<double> values;
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
          values.push_back(prediction.value.flat<float>()(i));
        }
        CHECK_EQ(actual_moves.size(), expected_moves.size());
        const double actual_value = good_ai == Color::kOne
                                        ? record.game_result()
                                        : -record.game_result();
        for (int i = 0; i < actual_moves.size(); ++i) {
          if (actual_moves[i] == expected_moves[i]) {
            ++correct;
          }
          total_value_loss += pow(values[i] - actual_value, 2);
          ++total;
        }
      }
      std::cout << "Accuracy: " << (100 * double(correct) / total) << "% \n";
      std::cout << "Avg value loss: " << (total_value_loss / total) << " \n";
    }
#endif
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
            Prediction pred;
            for (int i = 0; i < 7; ++i) {
              pred.move_p[i] = (i == move) ? 1.0 : 0.0;
            }
            pred.value = (board.turn() == Color::kOne) ? record.game_result()
                                                       : -record.game_result();
            trainer.Train(board, pred);
          }
          board.MakeMove(move);
        }
        CHECK(board.is_over());
      }
      trainer.Flush();
    }

    std::cout << "Saving checkpoint\n";
    model.Checkpoint(checkpoint_prefix);

  }
}

}  // namespace c4cc

int main(int argc, char* argv[]) {
  // Setup global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  c4cc::Go();
  return 0;
}
