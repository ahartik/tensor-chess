# load('@com_google_protobuf//:protobuf.bzl', 'proto_library')
# load('@com_google_protobuf//:protobuf.bzl', 'cc_proto_library')

proto_library(
    name = "game_proto",
    srcs = ["game.proto"],
)

cc_proto_library(
    name = "game_cc_proto",
    deps = [":game_proto"],
)

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
    hdrs = ["board.h", "board-inl.cpp"],
    deps = [
        ":bitboard",
        ":magic",
        ":game_cc_proto",
        "//util:int-set",
        "@com_google_absl//absl/strings",
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

cc_binary(
    name = "board_test_recordio",
    srcs = ["board_test_recordio.cpp"],
    deps = [
        ":board",
        ":game_cc_proto",
        "//util:recordio",
    ],
)

cc_binary(
    name = "board_move_tester",
    srcs = ["board_move_tester.cpp"],
    deps = [
        ":board",
    ],
)

cc_binary(
    name = "perft",
    srcs = ["perft.cpp"],
    deps = [
        ":board",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/base",
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
