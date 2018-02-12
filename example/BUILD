cc_binary(
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
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py"
    ],
)
