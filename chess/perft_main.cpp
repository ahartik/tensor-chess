#include <cstdlib>
#include <iostream>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "chess/board.h"
#include "chess/perft.h"

namespace chess {

const int64_t known_results[] = {
    1,       20,        400,        8902,        197281,
    4865609, 119060324, 3195901860, 84998978956, 2439530234167,
};

void Go(int d, Board b) {
  if (d >= int(sizeof(known_results) / sizeof(known_results[0]))) {
    std::cerr << "d too large: " << d << "\n";
    abort();
  }
  absl::Time start = absl::Now();
  int64_t p = Perft(b, d);
  absl::Time end = absl::Now();
  std::cout << p << "\n";
  std::cout << "Time: " << (end - start) << "\n";
  std::cout << "Leaves per second: "
            << int64_t(p / absl::ToDoubleSeconds(end - start))
            << "\n";
  if (b.board_hash() == Board().board_hash()) {
    if (p == known_results[d]) {
      std::cout << "Correct result\n";
    } else {
      std::cout << "Wrong result, expected " << known_results[d] << "\n";
    }
  }
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
