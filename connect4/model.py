import tensorflow as tf

class Model(object):
    NUM_MOVES = 7
    SHAPE = (2, 7 * 6)

    def add_conv2d(self, y, fx, fy, depth, stride=1):
        assert fx % 2 != 0
        assert fy % 2 != 0
        self.model_name += "_c{0}:{1}:{2}".format(fx, fy, depth)
        strides = [1, 1, stride, stride]

        y = tf.layers.conv2d(
            y,
            depth, (fx, fy),
            padding="same",
            data_format="channels_first"
            #, initializer=tf.contrib.layers.xavier_initializer()
        )
        y = tf.layers.batch_normalization(
            y, axis=1, training=self.is_training, fused=True)
        y = tf.nn.leaky_relu(y, alpha=0.01)
        return y

    def __init__(self, is_training, board, move_ind, result):
        self.model_name = "model"
        self.is_training = is_training
        self.move = tf.one_hot(move_ind, NUM_MOVES, dtype=tf.float32)

        # Smaller count leads to faster initial learning for sure.
        depth = 32

        # Add convolution layer(s)
        # conv_params = [(5, 5), (3, 3), (1, 8), (8, 1)]
        # Convert board to 2d format

        board_layers = int(board.shape[1])
        conv_output = tf.reshape(board, [-1, board_layers, 7, 6])

        num_layers = 12
        use_residual = True

        # SIMPLIFIED: just a few conv2d layers.
        conv_output = self.add_conv2d(conv_output, 3, 3, depth)
        conv_output = self.add_conv2d(conv_output, 3, 3, depth)

        # Convert each vertical slice to single value by doing a 2d convolution.
        cur_output = tf.layers.conv2d(
            cur_output,
            1, (1, 6),
            padding="valid",
            data_format="channels_first"
            #, initializer=tf.contrib.layers.xavier_initializer()
        )
        # No batchnorm here, as this is the output layer.
        cur_output = tf.reshape(cur_output, [-1, NUM_MOVES])

        logits = cur_output
        self.prediction = tf.nn.softmax(logits, name="prediction")

        # policy_loss_vec = -tf.reduce_sum(
        #     self.move * tf.log(tf.clip_by_value(self.prediction, 1e-8, 1.0)),
        #     axis=1)
        # policy_loss = tf.reduce_sum(policy_loss_vec)


        move_prob = tf.reduce_sum(self.prediction * self.move, axis=1)
        # If result is better than prediction, move was good - want bigger move
        # prob.
        self.reinforce_loss = -tf.reduce_sum(
            result * tf.log(tf.clip_by_value(move_prob, 1e-8, 1.0)))
        # self.reinforce_loss =  policy_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=0.001, use_locking=False).minimize(self.loss)

        self.reinforce_optimizer = tf.train.AdamOptimizer(
            learning_rate=0.001, use_locking=False,
            name="reinforce_optimizer").minimize(self.reinforce_loss)

        # Step 7: calculate accuracy with test set
        correct_preds = tf.equal(
            tf.argmax(self.prediction, 1), tf.argmax(self.move, 1))
        self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    def optimistic_restore(self, session, path):
        session.run(tf.global_variables_initializer())
        reader = tf.train.NewCheckpointReader(path)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0])
                            for var in tf.global_variables()
                            if var.name.split(':')[0] in saved_shapes]) restore_vars = []
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
                try:
                    curr_var = tf.get_variable(saved_var_name)
                except ValueError as e:
                    continue
                var_shape = curr_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(curr_var)
        saver = tf.train.Saver(restore_vars)
        saver.restore(session, path)
