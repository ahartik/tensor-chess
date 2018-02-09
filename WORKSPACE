workspace(name = "tensor-chess")

# To update TensorFlow to a new revision.
# 1. Update the 'git_commit' args below to include the new git hash.
# 2. Get the sha256 hash of the archive with a command such as...
#    curl -L https://github.com/tensorflow/tensorflow/archive/<git hash>.tar.gz | sha256sum
#    and update the 'sha256' arg with the result.
# 3. Request the new archive to be mirrored on mirror.bazel.build for more
#    reliable downloads.
tensorflow_http_archive(
  name = "org_tensorflow",
  sha256 = "21d6ac553adcfc9d089925f6d6793fee6a67264a0ce717bc998636662df4ca7e",
  git_commit = "bc69c4ceed6544c109be5693eb40ddcf3a4eb95d",
)

