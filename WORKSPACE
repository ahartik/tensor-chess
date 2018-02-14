workspace(name = "tensorchess")

local_repository(
    name = "org_tensorflow",
    path = __workspace_dir__ + "/tensorflow",
)


# TensorFlow depends on "io_bazel_rules_closure" so we need this here.
# Needs to be kept in sync with the same target in TensorFlow's WORKSPACE file.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "6691c58a2cd30a86776dd9bb34898b041e37136f2dc7e24cadaeaf599c95c657",
    strip_prefix = "rules_closure-08039ba8ca59f64248bb3b6ae016460fe9c9914f",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/08039ba8ca59f64248bb3b6ae016460fe9c9914f.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/08039ba8ca59f64248bb3b6ae016460fe9c9914f.tar.gz",  # 2018-01-16
    ],
)


local_repository(
  # Name of the Abseil repository. This name is defined within Abseil's
  # WORKSPACE file, in its `workspace()` metadata
  name = "com_google_absl",

  # NOTE: Bazel paths must be absolute paths. E.g., you can't use ~/Source
  path = __workspace_dir__ + "/abseil-cpp",
)

http_archive(
    name = "com_google_protobuf",
    urls = [
        "https://mirror.bazel.build/github.com/google/protobuf/archive/396336eb961b75f03b25824fe86cf6490fb75e3a.tar.gz",
        "https://github.com/google/protobuf/archive/396336eb961b75f03b25824fe86cf6490fb75e3a.tar.gz",
    ],
    sha256 = "846d907acf472ae233ec0882ef3a2d24edbbe834b80c305e867ac65a1f2c59e3",
    strip_prefix = "protobuf-396336eb961b75f03b25824fe86cf6490fb75e3a",
)

http_archive(
    name = "com_google_protobuf_cc",
    urls = [
        "https://mirror.bazel.build/github.com/google/protobuf/archive/396336eb961b75f03b25824fe86cf6490fb75e3a.tar.gz",
        "https://github.com/google/protobuf/archive/396336eb961b75f03b25824fe86cf6490fb75e3a.tar.gz",
    ],
    sha256 = "846d907acf472ae233ec0882ef3a2d24edbbe834b80c305e867ac65a1f2c59e3",
    strip_prefix = "protobuf-396336eb961b75f03b25824fe86cf6490fb75e3a",
)

# CCTZ (Time-zone framework), required by absl.
http_archive(
    name = "com_googlesource_code_cctz",
    urls = ["https://github.com/google/cctz/archive/master.zip"],
    strip_prefix = "cctz-master",
)


load('@org_tensorflow//tensorflow:workspace.bzl', 'tf_workspace')

tf_workspace(path_prefix = "", tf_repo_name = "org_tensorflow")
