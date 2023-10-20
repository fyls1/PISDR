import tensorflow as tf
import numpy as np

# exp: layer num [200, 100, 1] layer_acts = [tf.nn.relu, tf.nn.elu, None]
def get_dnn(x, layer_nums, layer_acts, name="dnn"):
    input_ft = x
    assert len(layer_nums) == len(layer_acts)
    with tf.variable_scope(name):
        for i, layer_num in enumerate(layer_nums):
            input_ft = tf.contrib.layers.fully_connected(
                inputs=input_ft,
                num_outputs=layer_num,
                scope='layer_%d' % i,
                activation_fn=layer_acts[i],
                reuse=tf.AUTO_REUSE)
    return input_ft

def get_dnn_critic(state, action, state_layer_nums, state_layer_acts, action_layer_nums, action_layer_acts, name="dnn"):
    input_ft = state
    assert len(state_layer_nums) == len(state_layer_acts)
    assert len(action_layer_nums) == len(action_layer_acts)
    with tf.variable_scope(name):
        for i, layer_num in enumerate(state_layer_nums):
            input_ft = tf.contrib.layers.fully_connected(
                inputs=input_ft,
                num_outputs=layer_num,
                scope='state_layer_%d' % i,
                activation_fn=state_layer_acts[i],
                reuse=tf.AUTO_REUSE)
        next_input_ft = tf.concat([input_ft, action], axis=-1)
        for i, layer_num in enumerate(action_layer_nums):
            next_input_ft = tf.contrib.layers.fully_connected(
                inputs=next_input_ft,
                num_outputs=layer_num,
                scope='action_layer_%d' % i,
                activation_fn=action_layer_acts[i],
                reuse=tf.AUTO_REUSE)
    return next_input_ft