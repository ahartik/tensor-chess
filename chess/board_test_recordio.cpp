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
  while (reader.Read(buf)) {
    MoveTestCase test_case;
    if (!test_case.ParseFromString(buf)) {
      std::cerr << "Invalid record\n";
      abort();
    }
    Board b(test_case.board());
    std::set<Move> expected_moves;
    for (const auto& m : test_case.valid_moves()) {
      expected_moves.insert(Move::FromProto(m));
    }
    const auto actual_moves_v = b.valid_moves();
    const std::set<Move> actual_moves(actual_moves_v.begin(),
                                      actual_moves_v.end());
    if (actual_moves != expected_moves) {
      std::cout << b.ToPrintString() << "\n";
      std::cout << b.ToFEN() << "\n";
      std::cout << "Missing moves:\n";
      // First log missing moves.
      for (const auto& m : expected_moves) {
        if (actual_moves.count(m) == 0) {
          std::cout << m.ToString() << ", ";
        }
      }
      std::cout << "\nIllegal moves:\n";
      for (const auto& m : actual_moves) {
        if (expected_moves.count(m) == 0) {
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
  chess::InitializeMovegen();
  for (int i = 1; i < argc; ++i) {
    std::cout << "Testing " << argv[i] << "\n";
    chess::Go(argv[i]);
  }
  return 0;
}
