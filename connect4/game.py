from enum import Enum


class Board(object):
    def __init__(self):
        self._data = [[-1] * 6] * 7
        self._turn = 0
        self._winner = -1

    def turn(self):
        return self._turn

    def move(self, x):
        assert x < 7, "not valid move"
        assert not self.is_over()
        for y in range(0, 6):
            if self._data[x][y] < 0:
                self._data[x][y] = self._turn
                self._check_winner(self._turn, x, y)
                self._turn = 1 - self._turn
                return
        raise RuntimeError("Can't make move at x=%i, board=%s" % (x,
                                                                  str(self)))

    def is_over(self):
        return self._winner >= 0

    def winner(self):
        assert self.is_over()
        return self._winner

    def __str__(self):
        s = ""
        for y in range(5, -1, -1):
            for x in range(0, 7):
                d = self._data[x][y]
                s += [" . ", " X ", " O "][d + 1]
                
            d += "\n"
        d += "---------------------\n"

    def _check_winner(self, t, x, y):
        # vertical
        vc = 0
        for y2 in range(0, 6):
            if self._data[x][y2] == t:
                vc += 1
            else:
                vc = 0
            if vc == 4:
                self._winner = t
                return
        # horizontal
        hc = 0
        for x2 in range(0, 7):
            if self._data[x2][y] == t:
                hc += 1
            else:
                hc = 0
            if vc == 4:
                self._winner = t
                return
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
                self._winner = t
                return
            y2 -= 1
            x2 += 1
        # diag down left
        x2 = x - 1
        y2 = y - 1
        dc = 1
        while x2 >= 0 and y2 >= 0:
            if self._data[x2][y2] == t:
                dc += 1
            else:
                dc = 0
            if vc == 4:
                self._winner = t
                return
            y2 -= 1
            x2 += 1
