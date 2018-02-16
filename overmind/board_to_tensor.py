import numpy as np
import chess
from board_pb2 import Board

HALFMOVE_LAYER = Board.NUM_LAYERS
REP_LAYER = Board.NUM_LAYERS + 1
NO_PROGRESS_LAYER = Board.NUM_LAYERS + 2

NUM_CHANNELS = Board.NUM_LAYERS + 3

empty_board_vec = np.array(NUM_CHANNELS * [[0.0] * 64], np.float32)

def encoded_to_tensor(board_str):
    board = Board.FromString(board_str)
    board_vec = empty_board_vec.copy()
    for x in range(0, Board.NUM_LAYERS):
        for j in range(0, 64):
            if (board.layers[x] >> j) & 1:
                board_vec[(x, j)] = 1.0
    for j in range(0, 64):
        board_vec[(HALFMOVE_LAYER, j)] = board.half_move_count * 0.01
        board_vec[(REP_LAYER, j)] = board.repetition_count * 0.01
        board_vec[(NO_PROGRESS_LAYER, j)] = board.no_progress_count * 0.01

    move = board.move_from * 64 + board.move_to
    if board.encoded_move_to >= 64:
        move = board.encoded_move_to * 64 + board.move_to
    return (board_vec, np.int32(move), np.float32(board.game_result))

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
def board_to_tensor(board, half_move_count, repetition_count = 1):
    # no flip here, caller must flip beforehand, as we don't know "black" moves.
    assert board.turn

    board_vec = empty_board_vec.copy()

    for i in range(0, 64):
        p = board.piece_at(i)
        if p != None:
            s = p.symbol()
            board_vec[(piece_to_idx[s], i)] = 1.0

    # en passant
    if board.ep_square:
        board_vec[(Board.OPP_EN_PASSANT, board.ep_square)] = 1.0

    # castling rights
    for x in range(0, 64):
        if (board.castling_rights >> x) & 1:
            if x < 8:
                board_vec[(Board.MY_CASTLE_RIGHTS, x)] = 1.0
            else:
                assert x >= 56
                board_vec[(Board.OPP_CASTLE_RIGHTS, x)] = 1.0

    # Legal moves
    for move in board.legal_moves:
        board_vec[(Board.MY_LEGAL_FROM, move.from_square)] = 1.0
        board_vec[(Board.MY_LEGAL_TO, move.to_square)] = 1.0

    for j in range(0, 64):
        board_vec[(HALFMOVE_LAYER, j)] = half_move_count * 0.01
        board_vec[(REP_LAYER, j)] = repetition_count * 0.01
        board_vec[(NO_PROGRESS_LAYER, j)] = board.halfmove_clock * 0.01

    # opponent legal moves (assuming null move from us)
    board.push(chess.Move.null())
    assert not board.turn
    for move in board.legal_moves:
        # This includes capture of king
        board_vec[(Board.OPP_LEGAL_FROM, move.from_square)] = 1.0
        board_vec[(Board.OPP_LEGAL_TO, move.to_square)] = 1.0
    board.pop()

    return board_vec
