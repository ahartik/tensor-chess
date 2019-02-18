cc_library(
    name = "board",
    srcs = ["board.cpp"],
    hdrs = ["board.h"],
    deps = [],
)

cc_library(
    name = "negamax",
    srcs = ["negamax.cpp"],
    hdrs = ["negamax.h"],
    deps = [":board"],
)

cc_library(
    name = "play_game",
    srcs = ["play_game.cpp"],
    hdrs = ["play_game.h"],
    deps = [":board"],
)

cc_binary(
    name = "human_play",
    srcs = ["human_play.cpp"],
    deps = [":board", ":play_game", ":negamax", "//util:init"],
)
