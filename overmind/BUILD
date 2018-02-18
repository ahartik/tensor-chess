load('@org_tensorflow//tensorflow:tensorflow.bzl', 'tf_kernel_library')
load('@org_tensorflow//tensorflow:tensorflow.bzl', 'tf_custom_op_library')

# load('@com_google_protobuf//protobuf.bzl', 'proto_library')
# load('@com_google_protobuf//protobuf.bzl', 'cc_proto_library')

tf_custom_op_library(
    name = "board_converter.so",
    srcs = ["board-converter.cpp"],
    deps = [
        ":board_cc_proto",
        "@com_google_absl//absl/synchronization",
    ],
    # linkopts=["-fsanitize=address"],
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
    data = [":board_converter.so"],
    tags = ["requires-gpu"],
    srcs_version = "PY3",
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py"
        ],
)

cc_library(
    name = "chess",
    srcs = ["chess.cc"],
    hdrs = ["chess.h"],
    deps = [
        ":board_cc_proto",
    ]
)

cc_library(
    name =  "persistent-int-map",
    hdrs = ["persistent-int-map.h"],
    deps = [
    ]
)

cc_test(
    name = "persistent-int-map_test",
    srcs = ["persistent-int-map_test.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        ":persistent-int-map",
        "@googletest//:gtest_main",
        ],
)
