Goal: Build a strong self-learning chess AI.

To get there:
  * Repository state must allow boot-strapping of learning
  * Try learning from puzzles
  * Build secondary AI for time control.
    => This will speed up learning, as important positions are studied more.

Basic time control:
  * Bigger the variance of following moves, the more time should be used.
    Variance can be measured as ternary entropy.
  * To save time, we can learn this function instead of computing it.

Ideas for smarter learning:
* Find positions with great variance => learn games starting from those
  positions.
* Learn from puzzles, like end game positions.
* Build AI to decide what learning method to use.

How to handle corner cases:
* Don't care about 50 move rule except during mcts ? MCTS will thus learn to
  avoid "drawy" positions and guess when it needs to do something.
* Include bit if 

Input layers:
12 Bitboards
1 en passant capture squares
1 rooks with castling rights
1 repetition count (for threefold repetition)
1 last moved from (bitboard, may be empty during training)
1 last moved to (bitboard, may be empty during training)

Network structure:
* 10-20 residual layers
* Each head ends in 1x1 convolution, potentially followed by dense layers.

Output "heads":
 * Policy: distribution with dims [73, 64]
   - Take [64, 64] slice from it and transpose it back.
 * Value: probability of win, probability of draw, probability of loss
 * Time usage learning: How much of remaining time should be used on this move.

Time usage learning:
* Uncertainty is measured by the difference between (win, loss) valuations
  between two consecutive moves. This is measured as entropy, used to weight
  time use: 50% of entropy gets 50% of time.
* To prevent local minima, randomly use 5-20% extra of remaining time on a
  move.
*


