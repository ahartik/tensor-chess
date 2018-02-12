from board_pb2 import Board

import tensorflow as tf

import chess
import chess.pgn
import lzma
import sys

fname = sys.argv[1]

piece_to_idx = {
    'P': Board.MY_P,
    'R': Board.MY_R,
    'B': Board.MY_B,
    'N': Board.MY_N,
    'Q': Board.MY_Q,
    'K': Board.MY_K,
    'p': Board.OPP_P,
    'r': Board.OPP_R,
    'b': Board.OPP_B,
    'n': Board.OPP_N,
    'q': Board.OPP_Q,
    'k': Board.OPP_K,
}

promo_to_layer = {
    chess.QUEEN: Board.MY_Q,
    chess.ROOK: Board.MY_R,
    chess.BISHOP: Board.MY_B,
    chess.KNIGHT: Board.MY_N,
}

underpromo_offset = {
    chess.ROOK: 0,
    chess.BISHOP: 1,
    chess.KNIGHT: 2,
}


def encode_underpromotion(from_square, to_square, promo):
    assert promo in [chess.ROOK, chess.BISHOP, chess.KNIGHT]
    if to_square == from_square + 8:
        return 64 + underpromo_offset[promo]
    if to_square == from_square + 7:
        return 67 + underpromo_offset[promo]
    if to_square == from_square + 9:
        return 70 + underpromo_offset[promo]
    raise RuntimeError("Bad promo: from {0} to {1}".format(
        from_square, to_square))


# Returns Board object
def encode_state(board, move_number, repetition_count, next_move, game_result):
    move_from_square =  next_move.from_square
    move_to_square =  next_move.to_square
    if board.turn:
        board = board.copy()
    else:
        def fixpos(i):
            rank = i // 8
            f = i % 8
            return (7 - rank) * 8 + f
        board = board.mirror()
        move_from_square = fixpos(next_move.from_square)
        move_to_square = fixpos(next_move.to_square)

    output = Board()
    output.layers.extend([0] * Board.NUM_LAYERS)

    for i in range(0, 64):
        p = board.piece_at(i)
        if p != None:
            s = p.symbol()
            # If it's black's turn, swap the board.
            if not board.turn:
                s = s.swapcase()
            output.layers[piece_to_idx[s]] |= 1 << i

    # en passant
    if board.ep_square:
        output.layers[Board.OPP_EN_PASSANT] |= 1 << board.ep_square

    # castling rights
    for x in range(0, 64):
        if (board.castling_rights >> x) & 1:
            if x < 8:
                output.layers[Board.MY_CASTLE_RIGHTS] |= 1 << x
            else:
                assert x >= 56
                output.layers[Board.OPP_CASTLE_RIGHTS] |= 1 << x

    # Legal moves
    for move in board.legal_moves:
        output.layers[Board.MY_LEGAL_FROM] |= 1 << move.from_square
        output.layers[Board.MY_LEGAL_TO] |= 1 << move.to_square

    # opponent legal moves (assuming null move from us)
    board.push(chess.Move.null())
    assert not board.turn
    for move in board.legal_moves:
        # This includes capture of king
        output.layers[Board.OPP_LEGAL_FROM] |= 1 << move.from_square
        output.layers[Board.OPP_LEGAL_TO] |= 1 << move.to_square
    board.pop()

    output.no_progress_count = board.halfmove_clock
    output.repetition_count = repetition_count
    output.half_move_count = move_number

    output.move_from = move_from_square
    output.move_to = move_to_square
    output.encoded_move_to = next_move.to_square
    if next_move.promotion != None:
        output.promotion = promo_to_layer[next_move.promotion]
        if next_move.promotion != chess.QUEEN:
            output.encoded_move_to = encode_underpromotion(
                output.move_from, output.move_to, next_move.promotion)

    output.game_result = game_result if board.turn else -game_result
    return output.SerializeToString()


pgn_file = open(sys.argv[1], "r")

result_to_score = {
    "1-0": 1,
    "1/2-1/2": 0,
    "0-1": -1,
}

count = 0

with tf.python_io.TFRecordWriter(sys.argv[2]) as output:
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
