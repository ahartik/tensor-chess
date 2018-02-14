load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

cc_binary(
    name = "example",
    srcs = ["example.cpp"] + if_cuda(["xyz"]),
    tags = ["requires-gpu"],
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
