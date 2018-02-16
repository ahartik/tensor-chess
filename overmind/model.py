import tensorflow as tf

NUM_MOVES = 64 * 73

class ChessFlow(object):
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
    
    
    def add_res_conv2d(self, y, fx, fy, mid_depth, out_depth, stride=1):
        strides = [1, 1, stride, stride]
        self.model_name += "_cr2{0}:{1}:{2}".format(fx, fy, mid_depth)
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
            y, axis=1, training=self.is_training, fused=True)
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
            y, axis=1, training=self.is_training, fused=True)
        y = tf.add(y, shortcut)
        y = tf.nn.leaky_relu(y, alpha=0.01)
        return y
    # 
    def __init__(self, is_training, board, move_ind, result):
        self.model_name = "model"
        self.is_training = is_training
        self.move = tf.one_hot(move_ind, NUM_MOVES, dtype=tf.float32)
        
        # 192 is copied from alphago, 128 might be enough, 256 could be better.
        # Smaller count leads to faster initial learning for sure.
        depth = 192
        
        # Add convolution layer(s)
        # conv_params = [(5, 5), (3, 3), (1, 8), (8, 1)]
        # Convert board to 2d format
        
        board_layers = int(board.shape[1])
        conv_output = tf.reshape(board, [-1, board_layers, 8, 8])
        
        num_layers = 12
        use_residual = True
        
        conv_output = self.add_conv2d(conv_output, 3, 3, depth)
        
        if use_residual:
            layers = 1
            while layers < num_layers - 1:
                conv_output = self.add_res_conv2d(conv_output, 3, 3, depth / 2, depth)
                layers += 2
            if layers < num_layers:
                conv_output = self.add_conv2d(conv_output, 3, 3, depth)
        else:
            for _ in range(0, num_layers - 1):
                conv_output = self.add_conv2d(conv_output, 7, 7, depth)
        
        cur_output = tf.reshape(conv_output, [-1, depth, 64])
        
        conv_output = cur_output
        
        NUM_MOVES_FROM = NUM_MOVES // 64
        
        cur_output = tf.layers.conv1d(
            cur_output,
            NUM_MOVES_FROM, (1, ),
            data_format="channels_first",
            use_bias=True)
        # No batchnorm here, as this is the output layer.
        
        cur_output = tf.reshape(cur_output, [-1, NUM_MOVES])
        
        logits = cur_output
        self.prediction = tf.nn.softmax(logits, name="prediction")
        
        policy_loss = -tf.reduce_sum(self.move * tf.log(tf.clip_by_value(self.prediction, 1e-8, 1.0)))
        
        #perform prediction for result
        
        # Take one filter from the last convolution output, only that will be used in
        # winner prediction.
        result_predict_input = tf.layers.conv1d(
            conv_output,
            1, (1, ),
            data_format="channels_last",
            use_bias=True,
            activation=tf.nn.leaky_relu)
        result_predict_input = tf.reshape(result_predict_input, [-1, depth])
        result_predict_hidden = tf.layers.dense(
            result_predict_input, 128, activation=tf.nn.leaky_relu)
        self.result_prediction = tf.layers.dense(
            result_predict_hidden, 1, activation=tf.nn.tanh)
        self.result_prediction = tf.reshape(self.result_prediction, [-1])
        self.result_loss = tf.reduce_sum(tf.squared_difference(self.result_prediction, result))
        
        self.loss = policy_loss * 1.00 + self.result_loss * 1.00
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=0.001, use_locking=False).minimize(self.loss)
        
        # Step 7: calculate accuracy with test set
        correct_preds = tf.equal(tf.argmax(self.prediction, 1),
                tf.argmax(self.move, 1))
        self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
    def save(self):
        pass
    def restore(self):
        pass
