#include "chess/tensors.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "chess/square.h"

namespace chess {
namespace {

constexpr int kNumMoves = 73 * 64;

TEST(MoveEncodingTest, SimpleMove) {
  Board b;
  const Move m = Move::FromString("e2e4").value();
  const int encoded = EncodeMove(b.turn(), m);
  EXPECT_LT(encoded, kNumMoves);
  EXPECT_EQ(DecodeMove(b, encoded), m);
}

TEST(MoveEncodingTest, BlackKnightMove) {
  Board b(Board(), Move::FromString("e2e4").value());
  // It's now blacks turn:
  ASSERT_EQ(b.turn(), Color::kBlack);

  const Move m = Move::FromString("b8c6").value();
  const int encoded = EncodeMove(b.turn(), m);
  EXPECT_LT(encoded, kNumMoves);
  EXPECT_EQ(DecodeMove(b, encoded), m);
}

// Tests encoding and decoding for all moves up to given depth. Returns the
// perft number: the number of leaves explored.
int64_t RecursivelyTestMoves(Board b, int depth) {
  if (depth == 0) {
    return 1;
  }
  int64_t sum = 0;
  for (const Move m : b.valid_moves()) {
    const int encoded = EncodeMove(b.turn(), m);
    EXPECT_LT(encoded, kNumMoves);
    EXPECT_EQ(DecodeMove(b, encoded), m) << "board:\n" << b.ToPrintString();
    sum += RecursivelyTestMoves(Board(b, m), depth - 1);
  }
  return sum;
}

TEST(MoveEncodingTest, RecursiveFromStart) {
  EXPECT_GT(RecursivelyTestMoves(Board(), 3), 10);
}

TEST(MoveEncodingTest, Promotions) {
  // Board where both white and black can promote in 3 directions. White to
  // move.
  Board b("r1b3n1/1P2k3/3p1bp1/8/5q2/8/1P2P1p1/1NBQKB1R w - - 0 1");

  // Promotion to knight in c8 is possible. This and other promotions will be
  // tested by the recursive call.
  Move m = Move::FromString("b7c8n").value();
  ASSERT_THAT(b.valid_moves(), ::testing::Contains(m));

  // Go up to depth 2, so that black's promotions are also checked.
  EXPECT_GT(RecursivelyTestMoves(b, 2), 10);
}

TEST(TensorConvertTest, BoardEncoding) {
  Board b;
  auto tensor = MakeBoardTensor(1);
  BoardToTensor(b, tensor.SubSlice(0));

  // Dimensions: batch, layer, square
  ASSERT_EQ(tensor.dims(), 3);
  auto access = tensor.tensor<float, 3>();

  // Check some (but not all) squares.

  // Layer 0 is pawns for the player whose turn it is.
  EXPECT_EQ(access(0, 0, Square::A2), 1.0);
  EXPECT_EQ(access(0, 0, Square::D3), 0.0);
  // Layer 5 is for my king.
  EXPECT_EQ(access(0, 5, Square::E1), 1.0);
  EXPECT_EQ(access(0, 5, Square::D1), 0.0);
  // Layer 6 is for opponent's pawns.
  EXPECT_EQ(access(0, 6, Square::C7), 1.0);
  // Layer 11 is for opponent's king.
  EXPECT_EQ(access(0, 11, Square::E8), 1.0);
  EXPECT_EQ(access(0, 11, Square::D8), 0.0);
}

}  // namespace
}  // namespace chess

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  chess::Board::Init();
  return RUN_ALL_TESTS();
}
