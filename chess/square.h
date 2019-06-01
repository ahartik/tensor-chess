#ifndef _CHESS_SQUARE_H_
#define _CHESS_SQUARE_H_

#include <cstdint>
#include <string>

namespace chess {

// Squares
class Square {
 public:
#define SQ(rank, rank_num)                         \
  static constexpr int rank##1 = 0 * 8 + rank_num; \
  static constexpr int rank##2 = 1 * 8 + rank_num; \
  static constexpr int rank##3 = 2 * 8 + rank_num; \
  static constexpr int rank##4 = 3 * 8 + rank_num; \
  static constexpr int rank##5 = 4 * 8 + rank_num; \
  static constexpr int rank##6 = 5 * 8 + rank_num; \
  static constexpr int rank##7 = 6 * 8 + rank_num; \
  static constexpr int rank##8 = 7 * 8 + rank_num

  SQ(A, 0);
  SQ(B, 1);
  SQ(C, 2);
  SQ(D, 3);
  SQ(E, 4);
  SQ(F, 5);
  SQ(G, 6);
  SQ(H, 7);
#undef SQ
  static int Rank(int sq) { return sq / 8; }
  static int File(int sq) { return sq % 8; }

  static std::string ToString(int sq);
};
static_assert(Square::A1 == 0);
static_assert(Square::H8 == 63);

inline int MakeSquare(int rank, int file) { return rank * 8 + file; }
inline int SquareRank(int pos) { return pos / 8; }
inline int SquareFile(int pos) { return pos % 8; }
inline bool SquareOnBoard(int rank, int file) {
  return rank >= 0 && rank < 8 && file >= 0 && file < 8;
}

}  // namespace chess

#endif
