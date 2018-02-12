load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")
load('@org_tensorflow//tensorflow:tensorflow.bzl', 'tf_kernel_library')

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

tf_kernel_library(
    name = "board-converter",
    srcs = ["board-converter.cpp"],
    deps = [
        ":board_cc_proto",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:client_session",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/memory",
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
