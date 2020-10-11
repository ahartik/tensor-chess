#include <iostream>
#include <string>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/optional.h"
#include "chess/board.h"
#include "chess/model.h"
#include "chess/player.h"
#include "chess/prediction_queue.h"
#include "chess/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

namespace chess {
namespace {

constexpr absl::string_view kName = "PolicyBot";
constexpr absl::string_view kAuthor = "Aleksi Hartikainen";

void FailWith(absl::string_view error) {
  std::cerr << error << "\n";
  std::cerr << "Exiting\n";
  abort();
}

class BotThread {
 public:
  BotThread() {
    CHECK(model_ != nullptr);
    player_ = std::make_unique<PolicyNetworkPlayer>(&pq_);
  }
  void NewGame() {}

  void SetPosition(const std::vector<std::string>& components) {
    Board pos;
    for (int i = 1; i < components.size(); ++i) {
      const std::string& c = components[i];
      if (c == "position" || c == "startpos") {
        continue;
      } else if (c == "fen") {
        // Start fen parsing.
        int end = components.size();
        for (int j = i + 1; j < components.size(); ++j) {
          if (components[j] == "moves") {
            end = j;
            break;
          }
        }
        pos = Board(absl::StrJoin(components.begin() + i + 1,
                                  components.begin() + end, " "));
        i = end;
      } else if (c.size() <= 5) {
        if (absl::optional<Move> m = Move::FromString(c)) {
          pos = Board(pos, *m);
        } else {
          FailWith("Invalid move string");
        }
      } else {
        // Assume FEN string.
        pos = Board(c);
      }
    }
    std::cerr << "Interpreted board:\n" << pos.ToPrintString() << "\n";
    std::cerr << pos.ToFEN() << "\n";
    player_->Reset(pos);
  }

  Move GetBestMove() const { return player_->GetMove(); }

 private:
  std::unique_ptr<Model> model_ = CreateDefaultModel(/*allow_init=*/false);
  PredictionQueue pq_{model_.get()};
  std::unique_ptr<Player> player_;
};  // namespace

void Go() {
  // First line should be "uci".
  std::string uciline;
  std::cin >> uciline;
  if (uciline != "uci") {
    FailWith(absl::StrCat("Expected 'uci', got '", uciline, "'"));
  }
  std::cout << "id name" << kName << "\n";
  std::cout << "id author " << kAuthor << "\n";
  // No supported options, they would be here.
  std::cout << "uciok\n";

  std::cin >> uciline;
  if (uciline != "isready") {
    FailWith(absl::StrCat("Expected 'isready', got '", uciline, "'"));
  }

  BotThread bot;

  std::cout << "readyok\n";

  while (std::getline(std::cin, uciline)) {
    std::cerr << "uciline: " << uciline << "\n";
    const std::vector<std::string> components = absl::StrSplit(uciline, " ");
    if (components[0] == "ucinewgame") {
      bot.NewGame();
    } else if (components[0] == "isready") {
      std::cout << "readyok\n";
    } else if (components[0] == "position") {
      bot.SetPosition(components);
    } else if (components[0] == "go") {
      if (std::count(components.begin(), components.end(), "ponder")) {
        continue;
      }
      std::cout << "bestmove " << bot.GetBestMove() << "\n";
    }
  }
}

}  // namespace
}  // namespace chess

int main(int argc, char** argv) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  chess::Go();
  return 0;
}
