import game_pb2

import chess
import chess.pgn

pgn_file = open(sys.argv[1], "r")

result_to_score = {
    "1-0": 1,
    "1/2-1/2": 0,
    "0-1": -1,
}

count = 0

def board_to_proto(b):
    proto = game_pb2.BoardProto()
    proto.

while True:
    game = chess.pgn.read_game(pgn_file)
    if game == None:
        break
    board = game.board()
    result_str = game.headers["Result"]
    if result_str not in result_to_score:
        continue
    result = result_to_score[result_str]

    move_number = 0
    repcounts = dict()
    for move in game.main_line():
        transpos_key = hash(board._transposition_key())
        repcount = repcounts[transpos_key] = repcounts.get(
            transpos_key, 0) + 1

        # Only write legal moves
        assert board.is_legal(move)
        output.write(
            encode_state(board, move_number, repcount, move, result))
        board.push(move)
        move_number += 1
    print(count)
    count += 1
