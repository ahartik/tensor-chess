load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_copts")

cc_library(
    name = "board",
    srcs = ["board.cpp"],
    hdrs = ["board.h"],
    deps = [],
)

cc_library(
    name = "negamax",
    srcs = ["negamax.cpp"],
    hdrs = ["negamax.h"],
    deps = [
        ":board",
        ":play_game",
    ],
)

cc_library(
    name = "play_game",
    srcs = ["play_game.cpp"],
    hdrs = ["play_game.h"],
    deps = [":board"],
)

cc_library(
    name = "human_player",
    srcs = ["human_player.cpp"],
    hdrs = ["human_player.h"],
    deps = [
        ":board",
        ":play_game",
    ],
)


proto_library(
    name = "game_proto",
    srcs = ["game.proto"],
)

cc_proto_library(
    name = "game_cc_proto",
    deps = [":game_proto"],
)

tf_cc_binary(
    name = "generate_ai_games",
    srcs = ["generate_ai_games.cpp"],
    deps = [
        ":board",
        ":negamax",
        ":game_cc_proto",
        ":play_game",
        "//util:init",
        "//util:recordio",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

cc_library (
    name = "model",
    hdrs = ["model.h"],
    srcs = ["model.cpp"],
    copts = tf_copts(),
    deps = [
        ":board",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

tf_cc_binary(
    name = "test_train",
    srcs = ["test_train.cpp"],
    deps = [
        ":board",
        ":model",
        ":game_cc_proto",
        "//util:recordio",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

cc_library(
    name = "predictor",
    hdrs = ["predictor.h"],
    srcs = ["predictor.cpp"],
    deps = [
        ":model",
        ":board",
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
        ":mcts",
        ":board",
        ":model",
        ":play_game",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

tf_cc_binary(
    name = "play",
    srcs = ["play.cpp"],
    deps = [
        ":board",
        ":human_player",
        ":model",
        ":mcts_player",
        ":negamax",
        ":play_game",
        "//util:init",
    ],
)
