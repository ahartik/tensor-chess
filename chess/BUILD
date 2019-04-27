cc_library(
    name = "bitboard",
    hdrs = ["bitboard.h"],
    deps = [
    ],
)

cc_test(
    name = "bitboard_test",
    srcs = ["bitboard_test.cpp"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        ":bitboard",
        "@googletest//:gtest_main",
    ],
)
