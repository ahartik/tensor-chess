import random
import glob
import numpy as np
import time
import os
import threading
import numpy as np
import board_to_tensor
import tensorflow as tf
import sys
import model

import board

np.set_printoptions(threshold=np.inf)

batch_size = 128
n_epochs = 3000

# Extra layers:
# - halfmove count
# - repetition count
# - no progress count

train_data = dataset_from_dir("/home/aleksi/tensor-chess-data/train")

test_data = dataset_from_dir("/home/aleksi/tensor-chess-data/test")

# dataset = dataset.shuffle(

is_training_tensor = tf.constant(False, dtype=tf.bool)
board_tensor = tf.placeholder(dtype=tf.float32, shape=(1, 2, 7 * 6))
maaaove_ind_tensor = tf.placeholder(dtype=tf.int32, shape=(1,1))
result_tensor = tf.constant(0, dtype=tf.float32)

train_iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                 train_data.output_shapes)
train_init = train_iterator.make_initializer(train_data)

test_iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                train_data.output_shapes)
test_init = test_iterator.make_initializer(test_data)

is_training = tf.placeholder(dtype=bool, shape=(), name="is_training")
board, move_ind, result = tf.cond(
    is_training,
    lambda: train_iterator.get_next(name="train_board"),
    lambda: test_iterator.get_next(name="test_board"))
cm = model.ChessFlow(is_training, board, move_ind, result)

writer = tf.summary.FileWriter('/home/aleksi/tensor-chess-data/graphs', tf.get_default_graph())
saver = tf.train.Saver()


def test_model(sess):
    # test the model first
    total_correct_preds = 0
    total_preds = 0
    total_loss = 0
    total_result_loss = 0
    sess.run(test_init)
    try:
        while total_preds < 10000:
            accuracy_batch, loss_batch, result_loss_batch = sess.run(
                [cm.accuracy, cm.loss, cm.result_loss], feed_dict={
                    is_training: False
                })
            total_preds += batch_size
            total_result_loss += result_loss_batch
            total_correct_preds += accuracy_batch
            total_loss += loss_batch
            assert (accuracy_batch <= batch_size)
    except tf.errors.OutOfRangeError:
        print("Ran out of test data")

    print('********* Test accuracy {0}'.format(
        total_correct_preds / total_preds))
    print('********* Test loss {0}'.format(total_loss / total_preds))
    print('********* Test result loss {0}'.format(total_result_loss / total_preds))
    sys.stdout.flush()


model_path = "/home/aleksi/tensor-chess-data/models/0/"
print("Model: {0}".format(cm.model_name))

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

graph_path = "/home/aleksi/tensor-chess-data/graphs/00"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    start_time = time.time()
    sess.run(train_init)

    if os.path.isdir(os.path.dirname(model_path)):
        saver.restore(sess, model_path)
    else:
        sess.run(tf.global_variables_initializer())

#     [b] = sess.run([board], feed_dict = {is_training: True})
#     print(b)
#     exit()


    test_model(sess)
    sys.stdout.flush()

    # train the model n_epochs times
    for epoch in range(n_epochs):
        last_log_time = start_time

        threads = 1
        items_per_step = 10000
        batches_per_thread = items_per_step // (threads * batch_size)
        items_per_step = batches_per_thread * threads * batch_size

        items_done = 0
        while items_done < 100000:
            try:
                train_threads = []
                def learn_func():
                    _, tl = sess.run([cm.optimizer, cm.loss], feed_dict = {is_training: True})
                    return tl
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
                print('Average loss so far in epoch {0} : {1}  {2} boards/s'.
                      format(epoch, total_loss / items_per_step,
                             items_per_step / time_since_log))
                saver.save(sess, model_path)
                items_done += items_per_step
                sys.stdout.flush()
            except tf.errors.OutOfRangeError:
                # Reinitialize
                print("Ran out of train data, reinitializing")
                sess.run(train_init)

        test_model(sess)
    print('Total time: {0} seconds'.format(time.time() - start_time))
    sys.stdout.flush()
    # test the model

    test_model(sess)
writer.close()
