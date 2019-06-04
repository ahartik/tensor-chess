#include <iostream>
#include <cstdlib>
#include <string>
#include <set>

#include "chess/board.h"
#include "absl/strings/string_view.h"
#include "chess/game.pb.h"
#include "util/recordio.h"

namespace chess {

void Go(absl::string_view fname) {
  util::RecordReader reader(fname);
  std::string buf;
  MoveTestCase test_case;
  std::vector<Move> expected_moves;
  while (reader.Read(buf)) {
    expected_moves.clear();
    if (!test_case.ParseFromString(buf)) {
      std::cerr << "Invalid record\n";
      abort();
    }
    Board b(test_case.board());
    for (const auto& m : test_case.valid_moves()) {
      expected_moves.push_back(Move::FromProto(m));
    }
    std::sort(expected_moves.begin(), expected_moves.end());
    auto gen_moves = b.valid_moves();
    std::sort(gen_moves.begin(), gen_moves.end());
    if (expected_moves.size() != gen_moves.size() ||
        !std::equal(gen_moves.begin(), gen_moves.end(),
                    expected_moves.begin())) {
      std::cout << b.ToPrintString() << "\n";
      std::cout << b.ToFEN() << "\n";
      std::cout << "ti should be " << int(b.turn()) << "\n";
      std::cout << "Missing moves:\n";
      std::set<Move> actual_moves(gen_moves.begin(), gen_moves.end());
      std::set<Move> expected_set(expected_moves.begin(), expected_moves.end());
      // First log missing moves.
      for (const auto& m : expected_moves) {
        if (actual_moves.count(m) == 0) {
          std::cout << m.ToString() << ", ";
        }
      }
      std::cout << "\nIllegal moves:\n";
      for (const auto& m : actual_moves) {
        if (expected_set.count(m) == 0) {
          std::cout << m.ToString() << ", ";
        }
      }
      std::cout << "\n";
      abort();
    }
  }
}

}  // namespace chess

int main(int argc, char** argv) {
  if (argc <= 1) {
    std::cerr << "Usage: " << argv[0] << " file.recordio\n";
    return 1;
  }
  chess::Board::Init();
  for (int i = 1; i < argc; ++i) {
    std::cout << "Testing " << argv[i] << "\n";
    chess::Go(argv[i]);
  }
  return 0;
}
