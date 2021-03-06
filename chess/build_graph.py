import tensorflow as tf

placeholder = tf.compat.v1.placeholder

keras = tf.keras

NUM_LAYERS = 14
MOVE_VEC_SIZE = 73 * 64


with tf.Graph().as_default() as graph:
# Batch of input and target output (1x1 matrices)
    board = placeholder(tf.float32, shape=[None, NUM_LAYERS, 64], name='board')
    target_move = placeholder(tf.float32, shape=[None, MOVE_VEC_SIZE], name='target_move')
# [p_win, p_loss, p_draw]
    target_value = placeholder(tf.float32, shape=[None], name='target_value')
    is_training = placeholder(dtype=bool, shape=(), name="is_training")

    layer = tf.reshape(board, shape=[-1, NUM_LAYERS, 8, 8], name='matrix')

# Add one layer of just values 1.0. Idea is that this helps convolutions know
# when they're at the edges of the boards when using SAME padding, as SAME
# padding zero-pads outside values. This theory hasn't been fully validated.
    paddings = tf.constant([[0,0], [0,1], [0,0], [0,0]])
    layer = tf.pad(layer, paddings, constant_values=1.0)

    print("Shape after pad: {}".format(layer.shape))

    channel_setup = 'channels_first'
    fused_bn = True
    num_filters = 64

    all_layers = []
    l2_reg = keras.regularizers.l2(0.001)

    def apply_layer(layer, y):
        all_layers.append(layer)
        return layer(y)


    def add_conv2d(y, res=True, bn=True, filters=num_filters, kernel=(3,3)):
        if not res:
            layer = tf.keras.layers.Conv2D(filters, kernel, padding='same',
                    data_format = channel_setup, kernel_regularizer=l2_reg)
            y = apply_layer(layer, y)
            if bn:
                y = tf.compat.v1.layers.batch_normalization(
                    y, axis=1, training=is_training, fused=fused_bn)
            y = tf.nn.leaky_relu(y, alpha=0.01)
        else:
            init = tf.keras.initializers.VarianceScaling(scale=0.1)
            shortcut = y
            layer = tf.keras.layers.Conv2D(filters, kernel, padding='same',
                    use_bias=False, data_format = channel_setup, kernel_regularizer=l2_reg,
                    kernel_initializer=init)
            y = apply_layer(layer, y)
            if bn:
                y = tf.compat.v1.layers.batch_normalization(
                    y, axis=1, training=is_training, fused=fused_bn)
            y = tf.nn.leaky_relu(y, alpha=0.01)

            # Another conv2d
            layer = tf.keras.layers.Conv2D(filters, kernel, padding='same',
                    use_bias=False, data_format = channel_setup, kernel_regularizer=l2_reg,
                    kernel_initializer=init)
            y = apply_layer(layer, y)
            if bn:
                y = tf.compat.v1.layers.batch_normalization(
                    y, axis=1, training=is_training, fused=fused_bn)
            y = tf.add(y, shortcut)
            y = tf.nn.leaky_relu(y, alpha=0.01)
        return y

# layer = add_conv2d(layer, res=False)
# layer = add_conv2d(layer, res=False)
    layer = add_conv2d(layer, res=False)

    num_blocks = 10

    for i in range(0, num_blocks):
        layer = add_conv2d(layer, res=True)

# Value head.
    print("Conv layer shape: {}".format(layer.shape));
    value_layer = add_conv2d(layer, res=False, filters=1, kernel=(1,1))
    value_layer = tf.keras.layers.Flatten()(value_layer)
    print("After flatten: {}".format(value_layer.shape));
    output_value = tf.keras.layers.Dense(1, activation='tanh')(value_layer)
    print("After dense: {}".format(value_layer.shape));
    output_value = tf.reshape(output_value, shape=[-1], name='output_value')


    if False:
        layer = tf.reshape(layer, [-1, num_filters, 64])
        
        top_slice = tf.slice(layer, [0, 0, 0], [-1, 64, -1])
        bottom_slice = tf.slice(layer, [0, 64, 0], [-1, num_filters - 64, -1])
        # Transpose top slice: 
        #  Before: batch, channel, rank, file
        #  After: batch, rank, file, channel
        print("top_slice: {}".format(top_slice.shape));
        top_slice = tf.transpose(top_slice, perm=[0, 2, 1])
        # Combine these together
        layer = tf.concat([top_slice, bottom_slice], 1)
        
        print("after concat: {}".format(layer.shape))

# Apply one more layer to the policy head
    layer = add_conv2d(layer, res=False)
    layer = tf.reshape(layer, [-1, num_filters, 64])
# Convert to 73 x 64 tensor that gets mapped as move prediction output.
# Basically each input filter is telling if this square is good to move to.
    layer = tf.compat.v1.layers.conv1d(layer, 73, 1, padding='valid', data_format='channels_first')
    print("Before policy head: {}".format(layer.shape))

    logits = tf.reshape(tensor=layer, shape=[-1, MOVE_VEC_SIZE], name = 'logits')

    output_move = tf.nn.softmax(logits, name='output_move')

# Optimize loss
    policy_loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
            labels=target_move, logits=logits, name='policy_loss')

    value_loss = tf.compat.v1.squared_difference(target_value, output_value,
            name='value_loss')

    l2_losses = [loss for layer in all_layers for loss in layer.losses]
    l2_loss = tf.math.add_n(l2_losses, name='l2_loss')
    print("l2_losses: {}".format(l2_losses))
    print("l2_loss.shape: {}".format(l2_loss.shape))

    total_loss = tf.add(1 * policy_loss, 1 * value_loss + 0.1 * l2_loss, name='total_loss')

    global train_op
# This weird 'with' thing is required for batch norm to work.
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.002, use_locking=True)
        train_op = optimizer.minimize(total_loss, name='train')

    init = tf.compat.v1.global_variables_initializer()

# tf.train.Saver.__init__ adds operations to the graph to save
# and restore variables.
    saver_def = tf.compat.v1.train.Saver().as_saver_def()

    print('Run this operation to initialize variables     : ', init.name)
    print('Run this operation for a train step            : ', train_op.name)
    print('Feed this tensor to set the checkpoint filename: ', saver_def.filename_tensor_name)
    print('Run this operation to save a checkpoint        : ', saver_def.save_tensor_name)
    print('Run this operation to restore a checkpoint     : ', saver_def.restore_op_name)

# Write the graph out to a file.
    with open('/mnt/tensor-data/chess-models/default/graph.pb', 'wb') as f:
      f.write(graph.as_graph_def().SerializeToString())

