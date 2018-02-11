load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

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

py_proto_library(
    name = "board-proto",
    srcs = ["board.proto"],
    deps = [],
)
