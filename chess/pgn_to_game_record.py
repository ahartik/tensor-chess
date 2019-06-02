import game_pb2

import sys
import chess
import chess.pgn

pgn_file = open(sys.argv[1], "r")

result_to_score = {
    "1-0": 1,
    "1/2-1/2": 0,
    "0-1": -1,
}

count = 0

piece_to_idx = {
    'P': 0,
    'N': 1,
    'B': 2,
    'R': 3,
    'Q': 4,
    'K': 5,
    'p': 6,
    'n': 7,
    'b': 8,
    'r': 9,
    'q': 10,
    'k': 11,
}

piece_to_white_idx = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

def board_to_proto(b, repetition_count):
    proto = game_pb2.BoardProto()
    proto.bitboards.extend(12 * [0])
    for i in range(0, 64):
        p = board.piece_at(i)
        if p != None:
            s = p.symbol()
            proto.bitboards[piece_to_idx[s]] |= 1 << i

    if b.ep_square:
        proto.en_passant = 1 << b.ep_square

    proto.castling_rights = board.castling_rights

    # Move counts
    proto.no_progress_count = board.halfmove_clock
    proto.repetition_count = repetition_count
    proto.half_move_count = (board.fullmove_number - 1) * 2 + (0 if board.turn else 1)

    return proto

def open_recordio(fname):
    # 'x' would fail if the file already exists
    f = open(fname, 'wb')
    # Super simple recordio version
    f.write(b'\x01')
    return f

def encode_le_u32(x):
    return bytes([x & 255,
        (x >> 8) & 255,
        (x >> 16) & 255,
        (x >> 24) & 255])

def append_record(f, proto):
    serialized = proto.SerializeToString()
    f.write(encode_le_u32(len(serialized)))
    f.write(serialized)

def append_moves(f, board):
    t = game_pb2.MoveTestCase()
    t.board.CopyFrom(board_to_proto(board, 0))
    for m in board.legal_moves:
        mp = t.valid_moves.add()
        mp.from_square = m.from_square
        mp.to_square = m.to_square
        if m.promotion:
            mp.promotion = piece_to_white_idx[m.promotion]
    append_record(f, t)

output_file = open_recordio(sys.argv[2])

while True:
    game = chess.pgn.read_game(pgn_file)
    # help(game)
    if game == None:
        break
    board = game.board()
    result_str = game.headers["Result"]
    if result_str not in result_to_score:
        continue
    record = game_pb2.GameRecord()

    record.result = result_to_score[result_str]
    for node in game.mainline():
        m = node.move

        mp = record.moves.add()
        mp.from_square = m.from_square
        mp.to_square = m.to_square
        if m.promotion:
            mp.promotion = piece_to_white_idx[m.promotion]

        board.push(m)

    append_record(output_file, record)



    print(count)
    count += 1
