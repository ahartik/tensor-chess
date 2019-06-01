#include <cstdlib>
#include <iostream>
#include <string>

#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "chess/board.h"

namespace chess {

const int64_t known_results[] = {
    1,       20,        400,        8902,        197281,
    4865609, 119060324, 3195901860, 84998978956, 2439530234167,

};

struct Action;
struct GameNode {
  GameNode() {}
  GameNode(const Board& p, const Move& m) : b(p, m) {}
  Board b;
  std::vector<Action> a;
};
struct Action {
  Move m;
  GameNode s;
};

int64_t Perft(const Board& parent, const Move& m, int d) {
  if (d <= 0) {
    return 1;
  }
  Board b(parent, m);
  int64_t nodes = 0;
#if 1
  if (d == 1) {
    b.LegalMoves([&](const Move& m) { ++nodes; });
  } else {
    b.LegalMoves([&](const Move& m) { nodes += Perft(b, m, d - 1); });
  }
#else
  const auto moves = b.valid_moves();
  for (const Move& m : moves) {
    nodes += Perft(b, m, d - 1);
  }
#endif
  return nodes;
}

int64_t Perft(const Board& b, int d) {
  if (d <= 0) {
    return 1;
  }
  int64_t nodes = 0;
  b.LegalMoves([&](const Move& m) { nodes += Perft(b, m, d - 1); });
  return nodes;
}

void Go(int d) {
  if (d >= (sizeof(known_results) / sizeof(known_results[0]))) {
    std::cerr << "d too large: " << d << "\n";
    abort();
  }
  absl::Time start = absl::Now();
  int64_t p = Perft(Board(), d);
  absl::Time end = absl::Now();
  std::cout << p << "\n";
  std::cout << "Time: " << (end - start) << "\n";
  std::cout << "Leaves per second: "
            << int64_t(p / absl::ToDoubleSeconds(end - start)) << "\n";
  if (p == known_results[d]) {
    std::cout << "Correct result\n";
  } else {
    std::cout << "Wrong result, expected " << known_results[d] << "\n";
  }
}

}  // namespace chess

int main(int argc, char** argv) {
  chess::InitializeMovegen();

  if (argc <= 1) {
    std::cerr << "Usage: " << argv[0] << " depth\n";
    return 1;
  }
  chess::Go(atoi(argv[1]));
  return 0;
}
