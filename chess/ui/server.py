#!/usr/bin/env python
import os

import chess
import urllib
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

static_dir = os.path.abspath("./static")


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
        san = urllib.parse.unquote(san)
        print("got board '{0}'".format(san))
        if san == "":
            board = chess.Board()
        else:
            board = chess.Board(san)
        predictions = []
        for m in board.legal_moves:
            predictions.append((m.uci(), 0.11111))
            if len(predictions) >= 10:
                break
        self.send_response(200)
        self.send_header('Content-type', 'text/json')
        self.end_headers()
        encoded = json.dumps(predictions)
        print("sending predictions '{0}'".format(encoded))
        self.wfile.write(bytes(encoded, "utf8"))

def run():
    print('starting server...')

    # Server settings
    # Choose port 8080, for port 80, which is normally used for a http server, you need root access
    server_address = ('127.0.0.1', 8090)
    httpd = HTTPServer(server_address, testHTTPServer_RequestHandler)
    print('running server...')
    httpd.serve_forever()


run()
