package(default_visibility = ["//visibility:public"])

cc_library(
    name = "init",
    srcs = ["init.cpp"],
    hdrs = ["init.h"],
    deps = [
        "@com_google_absl//absl/debugging:failure_signal_handler",
        "@com_google_absl//absl/debugging:symbolize",
    ],
)

cc_library(
    name = "recordio",
    srcs = ["recordio.cpp"],
    hdrs = ["recordio.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "recordio_test",
    srcs = ["recordio_test.cc"],
    deps = [
        ":recordio",
        "@googletest//:gtest_main",
    ],
)

cc_binary(
    name = "test_recordio",
    srcs = ["test_recordio.cpp"],
    deps = [
        ":recordio",
    ],
)

cc_library(
    name =  "refcount",
    hdrs = ["refcount.h"],
    deps = [
    ]
)

cc_library(
    name =  "persistent-int-map",
    hdrs = ["persistent-int-map.h"],
    deps = [
        ":refcount"
    ]
)

cc_test(
    name = "persistent-int-map_test",
    srcs = ["persistent-int-map_test.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        ":persistent-int-map",
        "@googletest//:gtest_main",
        "@com_google_absl//absl/time",
        ],
)

cc_library(
    name =  "int-set",
    hdrs = ["int-set.h"],
    srcs = ["int-set.cpp"],
    deps = [
        ":refcount"
    ]
)

cc_test(
    name =  "int-set_test",
    srcs = ["int-set_test.cpp"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        ":int-set",
        "@googletest//:gtest_main",
        "@com_google_absl//absl/time",
        ],
)
