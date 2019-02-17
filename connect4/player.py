import game
import random

def pick_human_move(board):
    legal = board.valid_moves()
    pstr = "X" if board.turn() == 0 else "O"
    while True:
        try:
            x = int(input("Enter move for " + pstr + ": "))
            if x in legal:
                return x
            else:
                print("Illegal move")
        except ValueError as e: 
            print(e)

def pick_random_move(board):
    legal = board.valid_moves()
    return legal[random.randrange(0, len(legal))]

# Returns -1 0 or 1
def human_play_game(p1, p2):
    b = game.Board()
    players = [p1, p2]
    while not b.is_over():
        print(b)
        m = players[b.turn()](b)
        b.move(m)
    print("Result: ", b.result())
    print("")
    return b.result()

def play_game(p1, p2):
    b = game.Board()
    players = [p1, p2]
    moves = []
    while not b.is_over():
        m = players[b.turn()](b)
        b.move(m)
        moves.append(m)
    return (b.result(), moves)
