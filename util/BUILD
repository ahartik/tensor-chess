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
  ],
)

cc_test(
    name =  "recordio_test",
    srcs = ["recordio_test.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        ":recordio",
        "@googletest//:gtest_main",
        ],
)

