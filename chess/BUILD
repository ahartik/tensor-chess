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
        "movegen.h",
    ],
    srcs = [
        "board.cpp",
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
        "@com_google_absl//absl/numeric:int128",
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
        "@com_google_absl//absl/types:optional",
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

cc_library(
    name = "player",
    hdrs = ["player.h"],
    deps = [
        ":board",
        ":types",
    ],
)

cc_library(
    name = "game_state",
    hdrs = ["game_state.h"],
    srcs = ["game_state.cpp"],
    deps = [
        ":board",
        ":types",
        "@com_google_absl//absl/container:flat_hash_map",
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

tf_cc_binary(
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
        "@com_google_absl//absl/strings:str_format",
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

cc_library(
    name =  "shuffling_trainer",
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

cc_library (
    name =  "prediction_queue",
    hdrs = ["prediction_queue.h"],
    srcs = ["prediction_queue.cpp"],
    copts = tf_copts(),
    deps = [
        ":board",
        ":model",
        ":tensors",
        ":prediction_cache",
        "@com_google_absl//absl/synchronization",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

cc_library (
    name =  "prediction_cache",
    hdrs = ["prediction_cache.h"],
   # srcs = [],
    deps = [
        ":board",
        ":types",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name =  "mcts",
    hdrs = ["mcts.h"],
    srcs = ["mcts.cpp"],
    copts = tf_copts(),
    deps = [
        ":board",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:optional",
        # This is only for logging, this library itself should not use
        # tensorflow for computation.
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

cc_library(
    name =  "mcts_player",
    hdrs = ["mcts_player.h"],
    srcs = ["mcts_player.cpp"],
    copts = tf_copts(),
    deps = [
        ":board",
        ":mcts",
        ":types",
        ":player",
        ":prediction_queue",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
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
        "//util:recordio",
    ],
)

tf_cc_binary(
    name = "play",
    srcs = ["play.cpp"],
    deps = [
        ":board",
        ":player",
        ":types",
        ":mcts",
        ":mcts_player",
        ":model",
        ":prediction_queue",
        ":game_state",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

tf_cc_binary(
    name =  "evaluate",
    srcs = ["evaluate.cpp"],
    deps = [
        ":board",
        ":player",
        ":types",
        ":mcts",
        ":mcts_player",
        ":model",
        ":prediction_queue",
        ":game_state",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

tf_cc_binary(
    name = "self_play_trainer",
    srcs = ["self_play_trainer.cpp"],
    deps = [
        ":board",
        ":player",
        ":types",
        ":magic",
        ":bitboard",
        ":mcts",
        ":mcts_player",
        ":shuffling_trainer",
        ":model",
        ":prediction_queue",
        ":prediction_cache",
        ":game_state",
        "//util:init",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

tf_cc_binary(
    name =  "uci_bot",
    srcs = ["uci_bot.cpp"],
    deps = [
        ":board",
        ":player",
        ":types",
        ":prediction_queue",
        ":model",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core:framework",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "movegen_test",
    srcs = ["movegen_test.cpp"],
    deps = [ ":board" ],
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
