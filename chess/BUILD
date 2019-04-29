cc_library(
    name = "bitboard",
    hdrs = ["bitboard.h"],
    deps = [
        "@com_google_absl//absl/strings",
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

cc_library(
    name = "board",
    hdrs = ["board.h"],
    srcs = ["board.cpp"],
    deps = [
        ":bitboard",
        "//util:int-set",
    ],
)

cc_library(
    name = "magic",
    hdrs = ["magic.h"],
    srcs = ["magic.cpp"],
    deps = [
        ":bitboard",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "magic_test",
    srcs = ["magic_test.cpp"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        ":magic",
        "@googletest//:gtest_main",
    ],
)

# cc_test(
#     name = "board_test",
#     srcs = ["board_test.cpp"],
#     copts = ["-Iexternal/gtest/include"],
#     deps = [
#         ":board",
#         "@googletest//:gtest_main",
#     ],
# )
