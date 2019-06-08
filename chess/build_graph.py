import tensorflow as tf

NUM_LAYERS = 14
MOVE_VEC_SIZE = 73 * 64

# Batch of input and target output (1x1 matrices)
board = tf.placeholder(tf.float32, shape=[None, NUM_LAYERS, 64], name='board')
target_move = tf.placeholder(tf.float32, shape=[None, MOVE_VEC_SIZE], name='target_move')
target_value = tf.placeholder(tf.float32, shape=[None], name='target_value')
is_training = tf.placeholder(dtype=bool, shape=(), name="is_training")

layer = tf.reshape(board, shape=[-1, NUM_LAYERS, 8, 8], name='matrix')

# Add one layer of just values 1.0. Idea is that this helps convolutions know
# when they're at the edges of the boards when using SAME padding, as SAME
# padding zero-pads outside values. This theory hasn't been fully validated.
paddings = tf.constant([[0,0], [0,1], [0,0], [0,0]])
layer = tf.pad(layer, paddings, constant_values=1.0)

print("Shape after pad: {}".format(layer.shape))


channel_setup = 'channels_first'
fused_bn = True

def add_conv2d(y, res=True, bn=True, filters=256, kernel=(3,3)):
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

num_blocks = 20

for i in range(0, num_blocks):
    layer = add_conv2d(layer, res=True)

# Value head.
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

# Convert to 73 x 8 x 8 tensor that gets mapped as move prediction output.
layer = tf.layers.conv2d(
    layer,
    73, (1, 1),
    data_format="channels_first",
    use_bias=True)
print("Before policy head: {}".format(layer.shape));

logits = tf.reshape(tensor=layer, shape=[-1, MOVE_VEC_SIZE], name = 'logits')

output_move = tf.nn.softmax(logits, name='output_move')

# Optimize loss
policy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=target_move, logits=logits, name='policy_loss')

value_loss = tf.squared_difference(target_value, output_value,
        name='value_loss')

total_loss = tf.add(policy_loss, value_loss, name='total_loss')

global train_op
# This weird 'with' thing is required for batch norm to work.
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.005, use_locking=True)
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
with open('/mnt/tensor-data/chess-models/default/graph.pb', 'wb') as f:
  f.write(tf.get_default_graph().as_graph_def().SerializeToString())

