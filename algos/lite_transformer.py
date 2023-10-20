#coding=utf-8
import tensorflow as tf
import numpy as np

def multihead_attention(queries, keys, values, num_heads, dropout_rate, training=True):
    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope("multihead_attention"):
        # Linear projections
        Q = tf.layers.dense(queries, d_model)
        K = tf.layers.dense(keys, d_model)
        V = tf.layers.dense(values, d_model)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        # Calculate attention
        outputs = scaled_dot_product_attention(Q_, K_, V_, dropout_rate, training)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        outputs += queries

        # Normalize
        outputs = layer_norm(outputs)
    return outputs

def scaled_dot_product_attention(Q, K, V, dropout_rate, training=True):
    d_k = Q.get_shape().as_list()[-1]
    outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
    outputs /= tf.sqrt(tf.cast(d_k, tf.float32))

    # Attention weights
    attention_weights = tf.nn.softmax(outputs)

    # Dropout
    attention_weights = tf.layers.dropout(attention_weights, rate=dropout_rate, training=training)
    outputs = tf.matmul(attention_weights, V)
    return outputs

def feed_forward(inputs, num_units):
    with tf.variable_scope("feed_forward"):
        # Inner layer
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

        # Outer layer
        outputs = tf.layers.dense(outputs, num_units[1])
        outputs += inputs

        # Normalize
        outputs = layer_norm(outputs)
    return outputs

def layer_norm(inputs, epsilon=1e-8):
    mean, var = tf.nn.moments(inputs, [-1], keep_dims=True)
    normalized = (inputs - mean) / tf.sqrt(var + epsilon)
    scale = tf.get_variable("layer_norm_scale", shape=[inputs.get_shape()[-1]], initializer=tf.ones_initializer())
    shift = tf.get_variable("layer_norm_shift", shape=[inputs.get_shape()[-1]], initializer=tf.zeros_initializer())
    return scale * normalized + shift

def positional_encoding(inputs, maxlen):
    E = inputs.get_shape().as_list()[-1]
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]
    with tf.variable_scope("positional_encoding"):
        position_ind = tf.tile(tf.expand_dims(tf.range(0, T), 0), [N, 1])
        position_enc = np.array([
            [pos / np.power(10000, 2.*i/E) for i in range(E)]
            for pos in range(maxlen)
        ])
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)
    return outputs

def decoder(inputs, maxlen, num_layers, num_heads, num_units, dropout_rate, training=True, name="decoder"):
    with tf.variable_scope(name):
        # Positional encoding
        inputs += positional_encoding(inputs, maxlen)

        for i in range(num_layers):
            with tf.variable_scope("decoder_layer_{}".format(i)):
                # Multi-head attention
                inputs = multihead_attention(inputs, inputs, inputs, num_heads, dropout_rate, training=training)

                # Feed-forward layer
                inputs = feed_forward(inputs, num_units)

    return inputs


if __name__ == '__main__':
    # 使用示例
    batch_size = 64
    seq_length = 20
    vocab_size = 8000
    d_model = 64
    num_layers = 2
    num_heads = 4
    dff = 128
    dropout_rate = 0.1

    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, shape=(batch_size, seq_length, d_model))
    output = decoder(inputs, seq_length, num_layers, num_heads, [dff, d_model], dropout_rate)

    # 创建Session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 构造虚拟输入数据
        input_data = np.random.randn(batch_size, seq_length, d_model)

        # 进行解码
        output_value = sess.run(output, feed_dict={inputs: input_data})

        # print(f"Decoder output shape: {output_value.shape}")
