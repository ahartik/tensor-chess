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
    sha256 = "43c9b882fa921923bcba764453f4058d102bece35a37c9f6383c713004aacff1",
    strip_prefix = "rules_closure-9889e2348259a5aad7e805547c1a0cf311cfcd91",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/9889e2348259a5aad7e805547c1a0cf311cfcd91.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/9889e2348259a5aad7e805547c1a0cf311cfcd91.tar.gz",  # 2018-12-21
    ],
)
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

