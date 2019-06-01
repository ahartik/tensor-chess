import chess
import random

while 1:
    b = chess.Board()
    while b.result(claim_draw=True) == '*':
        print(b.fen(en_passant='fen'))
        moves = list(b.legal_moves)
        ri = random.randint(0, len(moves) - 1)
        move = moves[ri];
        print(move.uci())
        b.push(moves[ri])
        print(b.fen(en_passant='fen'))
