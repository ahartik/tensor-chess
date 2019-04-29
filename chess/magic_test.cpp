#include "chess/magic.h"

#include <cstdint>
#include <vector>

#include "chess/bitboard.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace chess {
namespace {

constexpr absl::string_view kMyStr =
    "00111010\n"
    "01000000\n"
    "00100100\n"
    "00000000\n"
    "00000010\n"
    "00000000\n"
    "00000000\n"
    "00000000\n";
const uint64_t kMy = BitboardFromString(kMyStr);

constexpr absl::string_view kOppStr =
    "00000000\n"
    "00000000\n"
    "00000000\n"
    "00000000\n"
    "00000010\n"
    "00111000\n"
    "01010001\n"
    "10010011\n";
const uint64_t kOpp = BitboardFromString(kOppStr);

MATCHER_P(MatchesBitboard, str, "") {
  const uint64_t b = BitboardFromString(str);
  if (b != arg) {
    *result_listener << "Expected\n"
                     << str << "\n, got\n"
                     << BitboardToString(arg);
    return false;
  }
  return true;
}

TEST(MagicTest, Initialize) { InitializeMagic(); }

TEST(MagicTest, TestutilSanity) {
  EXPECT_THAT(kOpp, MatchesBitboard(kOppStr));
  EXPECT_THAT(kOpp, testing::Not(MatchesBitboard(kMyStr)));
}

TEST(MagicTest, Knight) {
  InitializeMagic();

  EXPECT_THAT(KnightMoveMask(MakeSquare(2, 2)), MatchesBitboard("01010000\n"
                                                                "10001000\n"
                                                                "00000000\n"
                                                                "10001000\n"
                                                                "01010000\n"
                                                                "00000000\n"
                                                                "00000000\n"
                                                                "00000000\n"));

  EXPECT_THAT(KnightMoveMask(MakeSquare(4, 0)), MatchesBitboard("00000000\n"
                                                                "00000000\n"
                                                                "01000000\n"
                                                                "00100000\n"
                                                                "00000000\n"
                                                                "00100000\n"
                                                                "01000000\n"
                                                                "00000000\n"));
}

TEST(MagicTest, BishopMoves) {
  InitializeMagic();
  EXPECT_THAT(
      BishopMoveMask(MakeSquare(2, 2), BitboardFromString("00111010\n"
                                                          "01000000\n"
                                                          "00100100\n"
                                                          "00000000\n"
                                                          "00000010\n"
                                                          "00000000\n"
                                                          "00000000\n"
                                                          "00000000\n")),
      MatchesBitboard("00001000\n"
                      "01010000\n"
                      "00000000\n"
                      "01010000\n"
                      "10001000\n"
                      "00000100\n"
                      "00000010\n"
                      "00000001\n"));

  EXPECT_THAT(
      BishopMoveMask(MakeSquare(4, 4), BitboardFromString("00111010\n"
                                                          "01000000\n"
                                                          "00100100\n"
                                                          "00000000\n"
                                                          "00001010\n"
                                                          "01000001\n"
                                                          "10110100\n"
                                                          "10100010\n")),
      MatchesBitboard("00000000\n"
                      "00000001\n"
                      "00100010\n"
                      "00010100\n"
                      "00000000\n"
                      "00010100\n"
                      "00100010\n"
                      "00000001\n"));
}

TEST(MagicTest, RookMoves) {
  InitializeMagic();
  EXPECT_THAT(RookMoveMask(MakeSquare(2, 2), BitboardFromString("00111010\n"
                                                                "01000000\n"
                                                                "00100100\n"
                                                                "00000000\n"
                                                                "00001010\n"
                                                                "01000001\n"
                                                                "10110100\n"
                                                                "10100010\n")),
              MatchesBitboard("00100000\n"
                              "00100000\n"
                              "11011100\n"
                              "00100000\n"
                              "00100000\n"
                              "00100000\n"
                              "00100000\n"
                              "00000000\n"));
}


}  // namespace
}  // namespace chess
