import tensorflow as tf

# Batch of input and target output (1x1 matrices)
board = tf.placeholder(tf.float32, shape=[None, 42 * 2], name='board')
target_move = tf.placeholder(tf.float32, shape=[None, 7], name='target_move')
target_value = tf.placeholder(tf.float32, shape=[None], name='target_value')
is_training = tf.placeholder(dtype=bool, shape=(), name="is_training")

layer = tf.reshape(board, shape=[-1, 2, 7, 6], name='matrix')

# Add one layer of just values 1.0. Idea is that this helps convolutions know
# when they're at the edges of the boards when using SAME padding, as SAME
# padding zero-pads outside values.
paddings = tf.constant([[0,0], [0,1], [0,0], [0,0]])
layer = tf.pad(layer, paddings, constant_values=1.0)

print("Shape after pad: {}".format(layer.shape))


channel_setup = 'channels_first'
fused_bn = True

def add_conv2d(y, res=True, bn=True, filters=64, kernel=(3,3)):
    if not res:
        y = tf.layers.conv2d(y, filters, kernel,
                data_format = channel_setup, padding='same')
        if bn:
            y = tf.layers.batch_normalization(
                y, axis=1, training=is_training, fused=fused_bn)
        y = tf.nn.leaky_relu(y, alpha=0.01)
    else:
        init = tf.variance_scaling_initializer(scale=0.1)
        shortcut = y
        y = tf.layers.conv2d(y, filters, kernel, data_format = channel_setup,
                use_bias=False, padding='same', kernel_initializer=init)
        if bn:
            y = tf.layers.batch_normalization(
                y, axis=1, training=is_training, fused=fused_bn)
        y = tf.nn.leaky_relu(y, alpha=0.01)
        y = tf.layers.conv2d(y, filters, kernel, data_format = channel_setup,
                use_bias=False, padding='same', kernel_initializer=init)
        if bn:
            y = tf.layers.batch_normalization(
                y, axis=1, training=is_training, fused=fused_bn)
        y = tf.add(y, shortcut)
        y = tf.nn.leaky_relu(y, alpha=0.01)
    return y

# layer = add_conv2d(layer, res=False)
# layer = add_conv2d(layer, res=False)
layer = add_conv2d(layer, res=False)
layer = add_conv2d(layer, res=True)
layer = add_conv2d(layer, res=True)
layer = add_conv2d(layer, res=True)
layer = add_conv2d(layer, res=True)
layer = add_conv2d(layer, res=True)
layer = add_conv2d(layer, res=True)
layer = add_conv2d(layer, res=True)

# Have one dense layer at this point 
print("Conv layer shape: {}".format(layer.shape));
value_layer = add_conv2d(layer, res=False, filters=1, kernel=(1,1))
value_layer = tf.keras.layers.Flatten()(value_layer)
print("After flatten: {}".format(value_layer.shape));
value_layer = tf.keras.layers.Dense(128, activation='relu')(value_layer)
print("After dense: {}".format(value_layer.shape));
value_layer = tf.keras.layers.Dense(128, activation='relu')(value_layer)
print("After dense: {}".format(value_layer.shape));
output_value = tf.keras.layers.Dense(1, activation='tanh')(value_layer)
print("After head: {}".format(output_value.shape));

output_value = tf.reshape(output_value, shape=[-1], name='output_value')

# Add one more layer for move prediction.
layer = add_conv2d(layer, res=True)

# Flatten to one value per column to get output.
layer = tf.layers.conv2d(layer, 1, (1, 6),
        data_format = channel_setup, padding='valid')

logits = tf.reshape(tensor=layer, shape=[-1, 7], name = 'logits')

output_move = tf.nn.softmax(logits, name='output_move')

# Optimize loss
policy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=target_move, logits=logits, name='policy_loss')

value_loss = tf.squared_difference(target_value, output_value)

total_loss = tf.add(policy_loss, value_loss, name='total_loss')

global train_op
# This weird 'with' thing is required for batch norm to work.
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(total_loss, name='train')

init = tf.global_variables_initializer()

# tf.train.Saver.__init__ adds operations to the graph to save
# and restore variables.
saver_def = tf.train.Saver().as_saver_def()

print('Run this operation to initialize variables     : ', init.name)
print('Run this operation for a train step            : ', train_op.name)
print('Feed this tensor to set the checkpoint filename: ', saver_def.filename_tensor_name)
print('Run this operation to save a checkpoint        : ', saver_def.save_tensor_name)
print('Run this operation to restore a checkpoint     : ', saver_def.restore_op_name)

# Write the graph out to a file.
with open('/mnt/tensor-data/c4cc/graph.pb', 'wb') as f:
  f.write(tf.get_default_graph().as_graph_def().SerializeToString())

