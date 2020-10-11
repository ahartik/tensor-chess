#include "chess/perft.h"

#include "chess/board.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace chess {
namespace {

TEST(MoveEncodingTest, StartingPos) {
  Board b;
  EXPECT_EQ(Perft(b, 1), 20);
  EXPECT_EQ(Perft(b, 2), 400);
  EXPECT_EQ(Perft(b, 3), 8902);
  EXPECT_EQ(Perft(b, 4), 197281);
  EXPECT_EQ(Perft(b, 5), 4865609);
  EXPECT_EQ(Perft(b, 6), 119060324);
}

TEST(MoveEncodingTest, Kiwipete) {
  // https://www.chessprogramming.org/Perft_Results#Position_2
  Board b(
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
  EXPECT_EQ(Perft(b, 1), 48);
  EXPECT_EQ(Perft(b, 2), 2039);
  EXPECT_EQ(Perft(b, 3), 97862);
  EXPECT_EQ(Perft(b, 4), 4085603);
  EXPECT_EQ(Perft(b, 5), 193690690);
}

TEST(MoveEncodingTest, Position3) {
  // https://www.chessprogramming.org/Perft_Results#Position_3
  Board b("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
  EXPECT_EQ(Perft(b, 1), 14);
  EXPECT_EQ(Perft(b, 2), 191);
  EXPECT_EQ(Perft(b, 3), 2812);
  EXPECT_EQ(Perft(b, 4), 43238);
  EXPECT_EQ(Perft(b, 5), 674624);
  EXPECT_EQ(Perft(b, 6), 11030083);
}

TEST(MoveEncodingTest, Position4) {
  // https://www.chessprogramming.org/Perft_Results#Position_4
  Board b("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
  EXPECT_EQ(Perft(b, 1), 6);
  EXPECT_EQ(Perft(b, 2), 264);
  EXPECT_EQ(Perft(b, 3), 9467);
  EXPECT_EQ(Perft(b, 4), 422333);
  EXPECT_EQ(Perft(b, 5), 15833292);
  EXPECT_EQ(Perft(b, 6), 706045033);
}

}  // namespace
}  // namespace chess

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  chess::Board::Init();
  return RUN_ALL_TESTS();
}
