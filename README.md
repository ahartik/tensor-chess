# Self-learning AI for chess and other games (abandoned)

This is a hobby project with a goal of creating a simple version of 
[AlphaZero](https://en.wikipedia.org/wiki/AlphaZero) self-learning algorithm
for chess.

Current status:
* Self-learning AI for connect 4 is very strong
* Same algorithm used for chess is not yet working, needs debugging
* Fast chess move generation: >200M moves/s (see chess/perft.cpp, [perft](https://www.chessprogramming.org/Perft))
* ML parts are very difficult to compile, requires compatible version combination
    of tensorflow, cuda, cudnn and gcc/clang.

