workspace(name = "tensorchess")

local_repository(
    name = "org_tensorflow",
    path = __workspace_dir__ + "/tensorflow",
)


local_repository(
  # Name of the Abseil repository. This name is defined within Abseil's
  # WORKSPACE file, in its `workspace()` metadata
  name = "com_google_absl",

  # NOTE: Bazel paths must be absolute paths. E.g., you can't use ~/Source
  path = __workspace_dir__ + "/abseil-cpp",
)
