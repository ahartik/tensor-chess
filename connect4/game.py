from enum import Enum
import numpy as np

class Board(object):
    _empty_tensor = np.array(2 * [[0.0] * 7 * 6], np.float32)
    def __init__(self):
        self._data = np.full((7, 6), -1, np.int8)
        self._turn = 0
        self._result = None

    def turn(self):
        return self._turn

    def move(self, x):
        assert x < 7, "not valid move"
        assert not self.is_over()
        for y in range(0, 6):
            if self._data[x][y] < 0:
                self._data[x][y] = self._turn
#                print("moved:\n")
#                print(self)
                self._check_winner(self._turn, x, y)
                self._turn = 1 - self._turn
                return
        raise RuntimeError("Can't make move at x=%i, board=%s" % (x,
                                                                  str(self)))

    def is_over(self):
        return self._result != None

    # 1 if player 0 won, -1 if player 1 won, 0 if draw.
    def result(self):
        assert self.is_over()
        return self._result

    def valid_moves(self):
        return [x for x in range(0, 7) if self._data[x][5] < 0]

    def to_tensor(self):
        tens = Board._empty_tensor.copy()
        pos = 0
        for x in range(0, 7):
            for y in range(0, 6):
                for c in range(0, 2):
                    if self._data[x][y] == (self._turn ^ c):
                        tens[(c, pos)] = 1.0
                pos += 1
        return tens

    def __str__(self):
        s = ""
        for y in range(5, -1, -1):
            for x in range(0, 7):
                d = self._data[x][y]
                s += [" . ", " X ", " O "][d + 1]
                
            s += "\n"
        s += "---------------------\n"
        s += " 0  1  2  3  4  5  6 \n"
        return s

    def _check_winner(self, t, x, y):
        # vertical
        vc = 0
        win_result = 1 if t == 0 else -1
        for y2 in range(0, 6):
            if self._data[x][y2] == t:
                vc += 1
            else:
                vc = 0
            if vc == 4:
                self._result = win_result
                return
        del vc
        # horizontal
        hc = 0
        for x2 in range(0, 7):
            if self._data[x2][y] == t:
                hc += 1
            else:
                hc = 0
            if hc == 4:
                self._result = win_result
                return
        del hc
        # diag down right
        x2 = x + 1
        y2 = y - 1
        dc = 1
        while x2 < 7 and y2 >= 0:
            if self._data[x2][y2] == t:
                dc += 1
            else:
                dc = 0
            if dc == 4:
                self._result = win_result
                return
            y2 -= 1
            x2 += 1
        # diag down left
        x2 = x - 1
        y2 = y - 1
        dc = 1
        while x2 < 7 and y2 >= 0:
            if self._data[x2][y2] == t:
                dc += 1
            else:
                dc = 0
            if dc == 4:
                self._result = win_result
                return
            y2 -= 1
            x2 += 1
        if len(self.valid_moves()) == 0:
            self._result = 0
