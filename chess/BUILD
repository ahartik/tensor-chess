# load('@com_google_protobuf//:protobuf.bzl', 'proto_library')
# load('@com_google_protobuf//:protobuf.bzl', 'cc_proto_library')
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_copts")

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
    hdrs = [
        "board.h",
        "board-inl.cpp",
    ],
    deps = [
        ":bitboard",
        ":game_cc_proto",
        ":magic",
        ":hashing",
        ":square",
        ":types",
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
        ":square",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "square",
    hdrs = ["square.h"],
    srcs = ["square.cpp"],
    deps = [ ],
)

cc_library(
    name = "types",
    srcs = ["types.cpp"],
    hdrs = ["types.h"],
    deps = [
        ":game_cc_proto",
        ":square",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "hashing",
    srcs = ["hashing.cpp"],
    hdrs = ["hashing.h"],
    deps = [
        ":types",
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
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library (
    name = "model",
    hdrs = ["model.h"],
    srcs = ["model.cpp"],
    copts = tf_copts(),
    deps = [
        "@com_google_absl//absl/synchronization",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

cc_library (
    name =  "tensors",
    hdrs = ["tensors.h"],
    srcs = ["tensors.cpp"],
    copts = tf_copts(),
    deps = [
        ":board",
        "@com_google_absl//absl/synchronization",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

cc_library (
    name =  "prediction_queue",
    hdrs = ["prediction_queue.h"],
    srcs = ["prediction_queue.cpp"],
    copts = tf_copts(),
    deps = [
        ":board",
        ":model",
        ":tensors",
        "@com_google_absl//absl/synchronization",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

cc_library(
    name = "shuffling_trainer",
    hdrs = ["shuffling_trainer.h"],
    srcs = ["shuffling_trainer.cpp"],
    copts = tf_copts(),
    deps = [
        ":board",
        ":tensors",
        ":model",
        "@com_google_absl//absl/synchronization",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

tf_cc_binary(
    name =  "tensor_test",
    srcs = ["tensor_test.cpp"],
    deps = [
        ":board",
        ":tensors",
        ":model",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

tf_cc_binary(
    name =  "human_imitation",
    srcs = ["human_imitation.cpp"],
    deps = [
        ":board",
        ":tensors",
        ":shuffling_trainer",
        ":model",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
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
