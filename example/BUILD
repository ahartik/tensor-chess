# load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "example",
    srcs = ["example.cpp"],
    deps = [
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:client_session",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

py_binary(
    name = "hello",
    srcs = ["hello.py"],
    tags = ["requires-gpu"],
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py"
    ],
)

tf_cc_binary(
    name = "model",
    srcs = [
        "model.cc",
        "data_set.h",
        "data_set.cc"
    ],
    deps = [
        "@org_tensorflow//tensorflow/cc:gradients",
        "@org_tensorflow//tensorflow/cc:grad_ops",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:client_session",
        "@org_tensorflow//tensorflow/core:tensorflow"
    ],
)
