import random
import chess
import glob
import time
import os
import threading
import numpy as np
import tensorflow as tf

from board_pb2 import Board

# Tensor shape
# 64 * 6 * 2 pieces
# 8 * 2 en_passant
# 2 * 2 castling
# [X, 64, 6
#

batch_size = 128
n_epochs = 3000

# Extra layers:
# - halfmove count
# - repetition count
# - no progress count

HALFMOVE_LAYER = Board.NUM_LAYERS
REP_LAYER = Board.NUM_LAYERS + 1
NO_PROGRESS_LAYER = Board.NUM_LAYERS + 2

NUM_CHANNELS = Board.NUM_LAYERS + 3

empty_board_vec = np.array(NUM_CHANNELS * [[0.0] * 64], np.float32)

NUM_MOVES = 64 * 73

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

    # print("move={0},{1}".format(move // 64, move % 64))
    move = board.encoded_move_to * 64 + board.move_from
    if board_vec[(Board.MY_LEGAL_FROM, board.move_from)] == 0.0:
        print("Bad move_from {0}, legal_from {1}".format(
            board.move_from, board.layers[Board.MY_LEGAL_FROM]))
        exit()
    if board_vec[(Board.MY_LEGAL_TO, board.move_to)] == 0.0:
        print("Bad move_to {0}, legal_to {1}".format(
            board.move_from, board.layers[Board.MY_LEGAL_TO]))
        exit()
    return (board_vec, np.int32(move), np.float32(board.game_result))


input_shape = (tf.TensorShape([NUM_CHANNELS * 64]), tf.TensorShape([]),
               tf.TensorShape([]))


def dataset_from_dir(path):
    filenames = glob.glob(os.path.join(path, "*.tfrecord"))
    random.shuffle(filenames)
    data = tf.data.TFRecordDataset(filenames)
    data = data.map(
            lambda tf_str :
            tuple(tf.py_func(encoded_to_tensor, [tf_str],
                [tf.float32, tf.int32, tf.float32]))
            )
    data = data.shuffle(10000)
    data = data.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    return data


train_data = dataset_from_dir("/home/aleksi/tensor-chess-data/train")

test_data = dataset_from_dir("/home/aleksi/tensor-chess-data/test")

# dataset = dataset.shuffle(

is_training = tf.placeholder(dtype=bool, shape=(), name="is_training")

train_iterator = tf.data.Iterator.from_structure(train_data.output_types,
        train_data.output_shapes)
train_init = train_iterator.make_initializer(train_data)

test_iterator = tf.data.Iterator.from_structure(train_data.output_types,
        train_data.output_shapes)
test_init = test_iterator.make_initializer(test_data)

board, move_ind, result = tf.cond(
    is_training,
    lambda: train_iterator.get_next(name="train_board"),
    lambda: test_iterator.get_next(name="test_board"))

move = tf.one_hot(move_ind, NUM_MOVES, dtype=tf.float32)

rand_init = tf.random_normal_initializer(0, 0.10)

model_name = "model"

conv_enabled = 1

def add_conv2d(y, fx, fy, depth, stride=1):
    assert fx % 2 != 0
    assert fy % 2 != 0
    global model_name
    model_name += "_c{0}:{1}:{2}".format(fx, fy, depth)
    strides = [1, 1, stride, stride]

    y = tf.layers.conv2d(
        y,
        depth, (fx, fy),
        padding="same",
        data_format="channels_first"
        #, initializer=tf.contrib.layers.xavier_initializer()
    )
    y = tf.layers.batch_normalization(
        y, axis=1, training=is_training, fused=True)
    y = tf.nn.leaky_relu(y, alpha=0.01)
    return y


def add_res_conv2d(y, fx, fy, mid_depth, out_depth, stride=1):
    global model_name
    strides = [1, 1, stride, stride]
    model_name += "_cr2{0}:{1}:{2}".format(fx, fy, mid_depth)
    shortcut = y

    init = None  # tf.variance_scaling_initializer(scale=0.1)

    y = tf.layers.conv2d(
        y,
        mid_depth, (fx, fy),
        padding="same",
        data_format="channels_first",
        use_bias=False,
        kernel_initializer=init)
    y = tf.layers.batch_normalization(
        y, axis=1, training=is_training, fused=True)
    y = tf.nn.leaky_relu(y, alpha=0.01)
    # Second layer
    y = tf.layers.conv2d(
        y,
        out_depth, (fx, fy),
        padding="same",
        data_format="channels_first",
        use_bias=False,
        kernel_initializer=init)
    y = tf.layers.batch_normalization(
        y, axis=1, training=is_training, fused=True)
    y = tf.add(y, shortcut)
    y = tf.nn.leaky_relu(y, alpha=0.01)
    return y


depth = 128
# Add convolution layer(s)
# conv_params = [(5, 5), (3, 3), (1, 8), (8, 1)]
# Convert board to 2d format
conv_output = tf.reshape(board, [-1, NUM_CHANNELS, 8, 8])

num_layers = 4
use_residual = True

conv_output = add_conv2d(conv_output, 3, 3, depth)

if use_residual:
    layers = 1
    while layers < num_layers - 1:
        conv_output = add_res_conv2d(conv_output, 3, 3, depth, depth)
        layers += 2
    if layers < num_layers:
        conv_output = add_conv2d(conv_output, 3, 3, depth)
else:
    for _ in range(0, num_layers - 1):
        conv_output = add_conv2d(conv_output, 7, 7, depth)

cur_output = tf.reshape(conv_output, [batch_size, depth, 64])

cur_output = tf.layers.conv1d(
    cur_output,
    NUM_MOVES // 64, (1, ),
    data_format="channels_first",
    use_bias=True)

cur_output = tf.reshape(cur_output, [batch_size, NUM_MOVES])

logits = cur_output
prediction = tf.nn.softmax(logits, name="prediction")

loss = -tf.reduce_sum(move * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(
        learning_rate=0.002, use_locking=True).minimize(loss)

# Step 7: calculate accuracy with test set
preds = tf.nn.softmax(logits)
# preds = my_softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(move, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
saver = tf.train.Saver()


def test_model(sess):
    # test the model first
    total_correct_preds = 0
    total_preds = 0
    total_loss = 0
    sess.run(test_init)
    try:
        while total_preds < 10000:
            total_preds += batch_size
            accuracy_batch, loss_batch = sess.run([accuracy, loss], feed_dict={is_training: False})
            total_correct_preds += accuracy_batch
            total_loss += loss_batch
            assert (accuracy_batch <= batch_size)
    except tf.errors.OutOfRangeError:
        print("Ran out of test data")

    print('********* Test accuracy {0}'.format(
        total_correct_preds / total_preds))
    print('********* Test loss {0}'.format(total_loss / total_preds))


model_path = "/home/aleksi/tensor-chess-data/models/0/"
print("Model: {0}".format(model_name))

cancelled = False


class TrainThread(threading.Thread):
    def __init__(self, batches, learn_func):
        self.total_loss = 0.0
        self.batches = batches
        self.learn_func = learn_func
        super(TrainThread, self).__init__()

    def run(self):
        for z in range(0, self.batches):
            global cancelled
            if cancelled:
                break
            try:
                self.total_loss += self.learn_func()
            except tf.errors.CancelledError:
                cancelled = True
                pass


with tf.Session() as sess:
    start_time = time.time()

    if os.path.isdir(os.path.dirname(model_path)):
        saver.restore(sess, model_path)
    else:
        sess.run(tf.global_variables_initializer())

    sess.run(train_init)
    test_model(sess)

    # train the model n_epochs times
    for epoch in range(n_epochs):
        last_log_time = start_time

        threads = 16
        items_per_step = 10000
        batches_per_thread = items_per_step // (threads * batch_size)
        items_per_step = batches_per_thread * threads * batch_size

        items_done = 0
        while items_done < 50000:
            try:
                train_threads = []
                learn_func = lambda: sess.run([optimizer, loss], feed_dict = {is_training: True})[1]
                for i in range(0, threads):
                    train_threads.append(
                        TrainThread(batches_per_thread, learn_func))
                for t in train_threads:
                    t.start()
                total_loss = 0
                for t in train_threads:
                    t.join()
                    total_loss += t.total_loss
                time_since_log = time.time() - last_log_time
                last_log_time += time_since_log
                print('Average loss so far in epoch {0}: {1}  {2} boards/s'.format(
                    epoch, total_loss / items_per_step,
                    items_per_step / time_since_log))
                saver.save(sess, model_path)
                items_done += items_per_step
            except tf.errors.OutOfRangeError:
                # Reinitialize
                print("Ran out of train data, reinitializing")
                sess.run(train_init)

        test_model(sess)
    print('Total time: {0} seconds'.format(time.time() - start_time))
    # test the model

    test_model(sess)
writer.close()
