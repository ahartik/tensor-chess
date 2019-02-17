import tensorflow as tf

class LearningPlayer(object):
    def __init__(self, model_dir):
        self.input_board = tf.placeholder(tf.float32, name="board")
        self.input_result = tf
        pass

    def pick_move(self, board):
        return 0

    def learn_games(self, games):
