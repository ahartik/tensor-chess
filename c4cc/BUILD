load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")

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
    deps = [":board"],
)

cc_library(
    name = "play_game",
    srcs = ["play_game.cpp"],
    hdrs = ["play_game.h"],
    deps = [":board"],
)

cc_binary(
    name = "human_play",
    srcs = ["human_play.cpp"],
    deps = [
        ":board",
        ":negamax",
        ":play_game",
        "//util:init",
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

tf_cc_binary(
    name = "test_train",
    srcs = ["test_train.cpp"],
    deps = [
        ":board",
        ":game_cc_proto",
        "//util:recordio",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)