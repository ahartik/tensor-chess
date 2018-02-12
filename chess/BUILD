load('@org_tensorflow//tensorflow:tensorflow.bzl', 'tf_kernel_library')
load('@org_tensorflow//tensorflow:tensorflow.bzl', 'tf_custom_op_library')

cc_library(
    name = "game-data",
    hdrs = ["game-data.h"],
    srcs = ["game-data.cpp"],
    deps = [
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:client_session",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/memory",
    ],
)

tf_custom_op_library(
    name = "board-converter.so",
    srcs = ["board-converter.cpp"],
    deps = [
        ":board_cc_proto",
    ],
)

proto_library(
    name = "board_proto",
    srcs = ["board.proto"],
)

cc_proto_library(
    name = "board_cc_proto",
    deps = [":board_proto"],
)

py_binary(
    name = "initial_trainer",
    srcs = ["initial_trainer.py"],
    data = [":board-converter.so"],
    srcs_version = "PY3",
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py"
        ],
)
