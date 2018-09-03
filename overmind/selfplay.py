import random
import glob
import chess
import numpy as np
import time
import os
import threading
import numpy as np
import board_to_tensor
import tensorflow as tf
import sys
import model
from board_pb2 import Board


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

def encode_move(move):
    if move.promotion != None:
        if move.promotion != chess.QUEEN:
            return encode_underpromotion(move.from_square, move.to_square, move.promotion)
    return move.from_square * 64 + move.to_square

def run_policy(sess, cm, board_tensor, board, half_move_count,
               repetition_count):
    orig_board = board

    def fixpos_nop(p):
        return p

    fixpos = fixpos_nop
    if not board.turn:

        def fixpos_black(i):
            rank = i // 8
            f = i % 8
            return (7 - rank) * 8 + f

        fixpos = fixpos_black
        board = board.mirror()

    bt = board_to_tensor.board_to_tensor(board, half_move_count,
                                         repetition_count)
    bt = np.reshape(bt, [1, bt.shape[0], bt.shape[1]])
    [pred_tensor] = sess.run([cm.prediction], {board_tensor: bt})

    predictions = []
    for move in board.legal_moves:
        prob = float(pred_tensor[(0, encode_move(move))])
        move.from_square = fixpos(move.from_square)
        move.to_square = fixpos(move.to_square)
        predictions.append((move, prob))

    return predictions

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


def encode_state(board, move_number, repetition_count, next_move, game_result):
    move_from_square = next_move.from_square
    move_to_square = next_move.to_square
    if board.turn:
        board = board.copy()
    else:
        game_result = -game_result

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

    output.game_result = game_result
    return output


def pick_random_move(board, policy):
    r = random.random()
    for (move, p) in policy:
        r -= p
        if r < 0:
            if board.is_legal(move):
                return move
            else:
                break
    # Failed: Return most probable legal move
    policy.sort(key=lambda x: x[1], reverse=True)
    for (move, p) in policy:
        if board.is_legal(move):
            return move

result_to_score = {
    "1-0": 1,
    "1/2-1/2": 0,
    "0-1": -1,
}

is_training_tensor = tf.constant(False, dtype=tf.bool)
board_tensor = tf.placeholder(
dtype=tf.float32, shape=(1, board_to_tensor.NUM_CHANNELS, 64))
move_ind_tensor = tf.constant(0, dtype=tf.int32)
result_tensor = tf.constant(0, dtype=tf.float32)

cm = model.ChessFlow(is_training_tensor, board_tensor, move_ind_tensor,
                 result_tensor)

def run_games(opp, num, output):
    model_path = "/home/aleksi/tensor-chess-data/models/next/"
    opp_path = os.path.join("/home/aleksi/tensor-chess-data/models/", str(opp))

    # Don't use all VRAM
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    sess_my = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    cm.optimistic_restore(sess_my, model_path)

    sess_opp = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    cm.optimistic_restore(sess_opp, opp_path)

    total_score = 0
    for game_no in range(0, num):
        # print("Game {}".format(game_no))
        board = chess.Board()
        player = random.randint(0, 1)
        sessions = [sess_my, sess_opp]
        move_number = 0
        repcounts = dict()
        states = []
        while not board.is_game_over(claim_draw=True):
            bot_idx = player if board.turn else 1 - player
            transpos_key = hash(board._transposition_key())
            repcount = repcounts[transpos_key] = repcounts.get(
                transpos_key, 0) + 1
            preds = run_policy(sessions[bot_idx], cm, board_tensor, board,
                               move_number, repcount)

            move = pick_random_move(board, preds)
            if move == None:
                print("No legal move")
                print(board)
                print("turn {}".format(board.turn))
                break
            # Only record moves by the learning player:
            if bot_idx == 0:
                states.append(encode_state(board, move_number, repcount, move, 0))

            board.push(move)
            move_number += 1
        score = result_to_score[board.result(claim_draw=True)]
        my_score = score if player == 0 else -score
        total_score += my_score
        ## print("move number {}".format(move_number))
        ## print(board)
        ## print("Result {}".format(score))
        score_char = {
                -1: "-",
                0 : ".",
                1: "+",
                }[my_score]
        print(score_char, end="")
        sys.stdout.flush()
        for x in states:
            x.game_result = score if player == 0 else -score
            output.write(x.SerializeToString())
    
    print("Avg. score {}".format(total_score / num))
    sys.stdout.flush()
    sess_my.close()
    sess_opp.close()

while True:
    output_path = "/home/aleksi/tensor-chess-data/selfplay/log_{}_{:08x}.tfrecord".format(
        time.strftime("%Y_%m_%d_%H:%M"), random.randint(0, (1 << 32) - 1))
    with tf.python_io.TFRecordWriter(output_path) as output:
        run_games("0/", 200, output)
