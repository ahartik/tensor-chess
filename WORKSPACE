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

# http_archive(
#     name = "com_google_protobuf",
#     urls = ["https://github.com/google/protobuf/archive/b4b0e304be5a68de3d0ee1af9b286f958750f5e4.zip"],
# )
# 
# # cc_proto_library rules implicitly depend on @com_google_protobuf_cc//:cc_toolchain,
# # which is the C++ proto runtime (base classes and common utilities).
# http_archive(
#     name = "com_google_protobuf_cc",
#     urls = ["https://github.com/google/protobuf/archive/b4b0e304be5a68de3d0ee1af9b286f958750f5e4.zip"],
# )


load('@org_tensorflow//tensorflow:workspace.bzl', 'tf_workspace')

tf_workspace(path_prefix = "", tf_repo_name = "org_tensorflow")
