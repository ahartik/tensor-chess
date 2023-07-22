workspace(name = "tensorchess")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

git_repository(
    name = "googletest",
    remote = "https://github.com/google/googletest",
    tag = "release-1.10.0",
)


# # Add tensorflow stuff after dependencies.
local_repository(
    name = "org_tensorflow",
    path = __workspace_dir__ + "/tensorflow",
)

# Initialize TensorFlow's external dependencies. 
load("@org_tensorflow//tensorflow:workspace3.bzl", "workspace")
workspace()
load("@org_tensorflow//tensorflow:workspace2.bzl", "workspace")
workspace() 
load("@org_tensorflow//tensorflow:workspace1.bzl", "workspace") 
workspace() 
load("@org_tensorflow//tensorflow:workspace0.bzl", "workspace") 
workspace() 



# # YCM stuff
# local_repository(
#     name = "compdb",
#     path = __workspace_dir__ + "/bazel-compilation-database",
# )

