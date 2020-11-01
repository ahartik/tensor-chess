# load('@com_google_protobuf//:protobuf.bzl', 'proto_library')
# load('@com_google_protobuf//:protobuf.bzl', 'cc_proto_library')
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_copts")

package(default_visibility = ["//visibility:public"])

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
    name =  "board",
    hdrs = ["board.h"],
    srcs = [],
    copts = tf_copts(),
    deps = [
        "@com_google_absl//absl/numeric:int128",
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
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

cc_library (
    name =  "shuffling_trainer",
    hdrs = ["shuffling_trainer.h"],
    srcs = ["shuffling_trainer.cpp"],
    copts = tf_copts(),
    deps = [
        ":board",
        ":model",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

cc_library (
    name =  "player",
    hdrs = ["player.h"],
    srcs = [],
    copts = tf_copts(),
    deps = [
        ":board",
    ],
)

cc_library (
    name =  "mcts",
    hdrs = ["mcts.h"],
    srcs = ["mcts.cpp"],
    copts = tf_copts(),
    deps = [
        ":board",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:optional",
        # This is only for logging, this library itself should not use
        # tensorflow for computation.
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

cc_library (
    name =  "mcts_player",
    hdrs = ["mcts_player.h"],
    srcs = ["mcts_player.cpp"],
    copts = tf_copts(),
    deps = [
        ":board",
        ":player",
        ":mcts",
        ":prediction_queue",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:optional",
        # This is only for logging, this library itself should not use
        # tensorflow for computation.
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

cc_library (
    name =  "prediction_cache",
    hdrs = ["prediction_cache.h"],
    srcs = ["prediction_cache.cpp"],
    deps = [
        ":board",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)
