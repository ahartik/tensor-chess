#include <cstdlib>
#include <iostream>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "chess/board.h"
#include "chess/movegen.h"

namespace chess {

const int64_t known_results[] = {
    1,       20,        400,        8902,        197281,
    4865609, 119060324, 3195901860, 84998978956, 2439530234167,

};

absl::flat_hash_map<uint64_t, int64_t> hashtable[10];
int64_t total_leaves = 0;

// Undef this to get easier-to-read profile output.
#define OPTIMIZED
// #define HASHED

int64_t Perft(const Board& parent, const Move& m, int d) {
  if (d <= 0) {
    return 1;
  }

  Board b(parent, m);
  int64_t nodes = 0;
#ifdef OPTIMIZED
  if (d == 1) {
    IterateLegalMoves(b, [&](const Move& m) {
      ++nodes;
      ++total_leaves;
    });
  } else {
    IterateLegalMoves(b,
                      [&](const Move& m) { nodes += Perft(b, m, d - 1); });
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
    IterateLegalMoves(b, [&](const Move& m) {
      ++nodes;
      ++total_leaves;
    });
  } else {
    IterateLegalMoves(
        b, [&](const Move& m) { nodes += PerftHashed(b, m, d - 1); });
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
#ifdef HASHED
    nodes += PerftHashed(b, m, d - 1);
#else
    nodes += Perft(b, m, d - 1);
#endif
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
            << int64_t(total_leaves / absl::ToDoubleSeconds(end - start))
            << "\n";
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
