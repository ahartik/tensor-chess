workspace(name = "tensorchess")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

# git_repository(
#     name = "googletest",
#     # build_file = "googletest/BUILD.bazel",
#     remote = "https://github.com/google/googletest",
#     tag = "release-1.8.1",
# )

# TensorFlow depends on "io_bazel_rules_closure" so we need this here.
# Needs to be kept in sync with the same target in TensorFlow's WORKSPACE file.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)
http_archive(
    name = "bazel_skylib",
    sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz"],
)  # https://github.com/bazelbuild/bazel-skylib/releases

# 
# # Add tensorflow stuff after dependencies.
local_repository(
    name = "org_tensorflow",
    path = __workspace_dir__ + "/tensorflow",
)
load('@org_tensorflow//tensorflow:workspace.bzl', 'tf_workspace')
tf_workspace(path_prefix = "", tf_repo_name = "org_tensorflow")
# 
# 
# # proto_library, cc_proto_library, and java_proto_library rules implicitly
# # depend on @com_google_protobuf for protoc and proto runtimes.
# # This statement defines the @com_google_protobuf repo.
# # http_archive(
# #     name = "com_google_protobuf",
# #     sha256 = "f976a4cd3f1699b6d20c1e944ca1de6754777918320c719742e1674fcf247b7e",
# #     strip_prefix = "protobuf-3.7.1",
# #     urls = ["https://github.com/protocolbuffers/protobuf/archive/v3.7.1.zip"],
# # )
# 
# # YCM stuff
# local_repository(
#     name = "compdb",
#     path = __workspace_dir__ + "/bazel-compilation-database",
# )

