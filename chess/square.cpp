#include "chess/square.h"

namespace chess {

std::string Square::ToString(int sq) {
  char buf[3] = {
      char('a' + (sq % 8)),
      char('1' + (sq / 8)),
      0,
  };
  return buf;
}

}
