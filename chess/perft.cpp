#include <cstdlib>
#include <iostream>
#include <string>

#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "chess/board.h"
#include "absl/container/flat_hash_map.h"

namespace chess {

const int64_t known_results[] = {
    1,       20,        400,        8902,        197281,
    4865609, 119060324, 3195901860, 84998978956, 2439530234167,

};

absl::flat_hash_map<uint64_t, int64_t> hashtable[10];

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

// Undef this to get easier-to-read profile output.
#define OPTIMIZED

int64_t Perft(const Board& parent, const Move& m, int d) {
  if (d <= 0) {
    return 1;
  }

  Board b(parent, m);
  int64_t nodes = 0;
#ifdef OPTIMIZED
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

int64_t hash_skips = 0;
int64_t PerftHashed(const Board& parent, const Move& m, int d) {
  if (d <= 0) {
    return 1;
  }

  Board b(parent, m);
  int64_t& nodes = hashtable[d][b.board_hash()];
  if (nodes > 0) {
    ++hash_skips;
    return nodes;
  }
  if (d == 1) {
    b.LegalMoves([&](const Move& m) { ++nodes; });
  } else {
    b.LegalMoves([&](const Move& m) { nodes += PerftHashed(b, m, d - 1); });
  }
  return nodes;
}


int64_t Perft(const Board& b, int d) {
  if (d <= 0) {
    return 1;
  }
  int64_t nodes = 0;
  const auto moves = b.valid_moves();
  for (const Move& m : moves) {
    nodes += PerftHashed(b, m, d - 1);
  }
  return nodes;
}

void Go(int d, Board b) {
  if (d >= (sizeof(known_results) / sizeof(known_results[0]))) {
    std::cerr << "d too large: " << d << "\n";
    abort();
  }
  absl::Time start = absl::Now();
  int64_t p = Perft(b, d);
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
  std::cout << "hash_skips: " << hash_skips << "\n";
}

}  // namespace chess

int main(int argc, char** argv) {
  chess::Board::Init();

  if (argc <= 1) {
    std::cerr << "Usage: " << argv[0] << " depth\n";
    return 1;
  }
  chess::Board b;
  if (argc > 2) {
    b = chess::Board(argv[2]);
  }

  chess::Go(atoi(argv[1]), b);
  return 0;
}
