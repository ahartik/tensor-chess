load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_copts")
# load(
#     "@org_tensorflow//tensorflow/core:platform/default/build_config.bzl",
#     "tf_proto_library",
# )

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
        "@com_google_absl//absl/synchronization",
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
    name =  "mcts_player",
    hdrs = ["mcts_player.h"],
    srcs = ["mcts_player.cpp"],
    copts = tf_copts(),
    deps = [
        ":board",
        ":generic_board",
        ":play_game",
        "//generic:mcts",
        "//generic:prediction_queue",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

cc_library(
    name = "prediction_queue",
    hdrs = ["prediction_queue.h"],
    srcs = ["prediction_queue.cpp"],
    copts = tf_copts(),
    deps = [
        ":board",
        ":board_tensor",
        "//generic:model",
        "@com_google_absl//absl/synchronization",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

cc_library(
    name = "shuffling_trainer",
    hdrs = ["shuffling_trainer.h"],
    srcs = ["shuffling_trainer.cpp"],
    copts = tf_copts(),
    deps = [
        ":board",
        ":board_tensor",
        "//generic:model",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

tf_cc_binary(
    name = "play",
    srcs = ["play.cpp"],
    deps = [
        ":board",
        ":model_collection",
        ":human_player",
        ":mcts_player",
        ":negamax",
        ":play_game",
        "//generic:prediction_queue",
        "//util:init",
        "@com_google_absl//absl/synchronization",
    ],
)

tf_cc_binary(
    name = "self_play_trainer",
    srcs = ["self_play_trainer.cpp"],
    deps = [
        ":board",
        ":generic_board",
        ":mcts_player",
        ":model_collection",
        ":play_game",
        "//generic:board",
        "//generic:model",
        "//generic:shuffling_trainer",
        "//util:init",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

tf_cc_binary(
    name =  "evaluate",
    srcs = ["evaluate.cpp"],
    deps = [
        ":board",
        ":model",
        ":mcts_player",
        ":play_game",
        "//util:init",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "board_tensor",
    srcs = ["board_tensor.cpp"],
    hdrs = ["board_tensor.h"],
    copts = tf_copts(),
    deps = [
        ":board",
        "//generic:model",
    ],
)

cc_library(
    name = "model_collection",
    srcs = ["model_collection.cpp"],
    hdrs = ["model_collection.h"],
    copts = tf_copts(),
    deps = [
        "//generic:model",
    ],
)

cc_library(
    name =  "generic_board",
    srcs = ["generic_board.cpp"],
    hdrs = ["generic_board.h"],
    copts = tf_copts(),
    deps = [
        ":board",
        "//generic:board",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/hash",
    ],
)
