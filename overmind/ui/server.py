#!/usr/bin/env python

import sys
sys.path.append("..")
import os
import tensorflow as tf
import chess
import urllib
import json
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer

import board_to_tensor
import model

static_dir = os.path.abspath("./static")
model_path = "/home/aleksi/tensor-chess-data/models/0/"

is_training_tensor = tf.constant(False, dtype=tf.bool)
board_tensor = tf.placeholder(dtype=tf.float32, shape=(1, board_to_tensor.NUM_CHANNELS, 64))
move_ind_tensor = tf.constant(0, dtype=tf.int32)
result_tensor = tf.constant(0, dtype=tf.float32)

cm = model.ChessFlow(is_training_tensor, board_tensor, move_ind_tensor, result_tensor)
saver = tf.train.Saver()

# Don't use all VRAM
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
saver.restore(sess, model_path)

def run_policy(board, half_move_count, repetition_count):
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
    for fq in range(0, 64):
        for tq in range(0, 64):
            m = chess.Move(fixpos(fq), fixpos(tq))
            if orig_board.is_legal(m):
                predictions.append((m.uci(), float(pred_tensor[(0, fq * 64 + tq)])))
    predictions.sort(key=lambda x: x[1], reverse=True)
    predictions = predictions[0:5]
    return predictions

# HTTPRequestHandler class
class testHTTPServer_RequestHandler(BaseHTTPRequestHandler):
    def send_my_header(self, filename):
        if filename[-4:] == '.css':
            self.send_header('Content-type', 'text/css')
        elif filename[-5:] == '.json':
            self.send_header('Content-type', 'application/javascript')
        elif filename[-3:] == '.js':
            self.send_header('Content-type', 'application/javascript')
        elif filename[-4:] == '.ico':
            self.send_header('Content-type', 'image/x-icon')
        else:
            self.send_header('Content-type', 'text/html')
        self.end_headers()
    # GET
    def do_GET(self):
        if self.path.startswith("/predict/"):
            self.do_predict(self.path.split("/", 2)[2])
            return
        # Send message back to client
        # Write content as utf-8 data
        print("path='{0}'".format(self.path))
        if self.path == "/":
            self.path = "/index.html"
        path = os.path.abspath("./static" + self.path)
        if not path.startswith(static_dir):
            # Send response status code
            self.send_response(403)
            # Send headers
            self.send_header('Content-type', 'text/html')
            self.wfile.write(bytes("VERBOTEN!", "utf8"))
            return

        try:
            with open(path, "rb") as f:
                self.send_response(200)
                self.send_my_header(path)
                static_page = f.read()
                self.wfile.write(static_page)
        except FileNotFoundError as e:
            print(e)
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"Not found :(")
        return
    def do_predict(self, san):
        [fen, half_moves] = json.loads(urllib.parse.unquote(san))
        print("got board '{0}' with {1} half moves so far".format(fen, half_moves))
        if san == "":
            board = chess.Board()
        else:
            board = chess.Board(fen)
        predictions = run_policy(board, half_moves, 1)

        self.send_response(200)
        self.send_header('Content-type', 'text/json')
        self.end_headers()
        encoded = json.dumps(predictions)
        self.wfile.write(bytes(encoded, "utf8"))

def run():

    # Server settings
    # Choose port 8080, for port 80, which is normally used for a http server, you need root access
    server_address = ('0.0.0.0', 8090)
    print('starting server on {0}...'.format(server_address))
    httpd = HTTPServer(server_address, testHTTPServer_RequestHandler)
    print('running server...')
    httpd.serve_forever()


run()
