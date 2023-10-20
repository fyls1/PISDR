#coding=utf-8
import datetime

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl as core_rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
import sys
from utils import *

def build_embedding(self, user_spare_feature, item_spare_feature, hist_spare_feature, args=None):
    used_user_feature = {"user_id": 0}
    used_item_feature = {"item_id": 0, "cat_id": 1, "cat1_id": 2, "shop_id": 3, "price_id": 4}
    used_hist_feature = {"item_id": 0, "cat_id": 1, "cat1_id": 2, "shop_id": 3, "price_id": 4}
    # 创建Embedding
    uid_embeddings_var = tf.get_variable("uid_embedding_var", [args.user_size, args.user_embedding_size])
    uid_batch_embedded = tf.nn.embedding_lookup(uid_embeddings_var, user_spare_feature[:, 0])

    mid_embeddings_var = tf.get_variable("mid_embedding_var", [args.item_size, args.item_embedding_size])
    mid_batch_embedded = tf.nn.embedding_lookup(mid_embeddings_var, item_spare_feature[:, :, 0])
    mid_his_batch_embedded = tf.nn.embedding_lookup(mid_embeddings_var, hist_spare_feature[:, :, 0])

    cat_embeddings_var = tf.get_variable("cat_embedding_var", [args.cate_size, args.cate_embedding_size])
    cat_batch_embedded = tf.nn.embedding_lookup(mid_embeddings_var, item_spare_feature[:, :, 1])
    cat_his_batch_embedded = tf.nn.embedding_lookup(mid_embeddings_var, hist_spare_feature[:, :, 1])

    cat1_embeddings_var = tf.get_variable("cat1_embedding_var", [args.cate1_size, args.cate1_embedding_size])
    cat1_batch_embedded = tf.nn.embedding_lookup(mid_embeddings_var, item_spare_feature[:, :, 2])
    cat1_his_batch_embedded = tf.nn.embedding_lookup(mid_embeddings_var, hist_spare_feature[:, :, 2])

    shop_embeddings_var = tf.get_variable("shop_embedding_var", [args.shop_size, args.shop_embedding_size])
    shop_batch_embedded = tf.nn.embedding_lookup(mid_embeddings_var, item_spare_feature[:, :, 3])
    shop_his_batch_embedded = tf.nn.embedding_lookup(mid_embeddings_var, hist_spare_feature[:, :, 3])

    price_embeddings_var = tf.get_variable("price_embedding_var", [args.price_size, args.price_embedding_size])
    price_batch_embedded = tf.nn.embedding_lookup(mid_embeddings_var, item_spare_feature[:, :, 4])
    price_his_batch_embedded = tf.nn.embedding_lookup(mid_embeddings_var, hist_spare_feature[:, :, 4])

class SLCbuSessionActorOneHot:
    """The Supversived Actor"""
    def __init__(self, user_spare_num=8, user_dense_num=14, item_num=12, item_spare_num=37, item_dense_num=29, hist_num=62, hist_sapre_num=7, num_actions=12, max_action=12, name="SL_CBU_Session", args=None):
        self._name = name
        self.state_dim = args.user_embedding_size + user_dense_num + (item_num+hist_num) * (args.item_embedding_size + args.cate_embedding_size + args.cate1_embedding_size + \
                                                                                            args.shop_embedding_size + args.price_embedding_size) + item_num * item_dense_num

        with tf.variable_scope(self._name):
            self.usr_spar_ph = tf.placeholder(tf.int32, [None, user_spare_num], name='user_spar')
            self.usr_dens_ph = tf.placeholder(tf.float32, [None, user_dense_num], name='user_dense')
            self.itm_spar_ph = tf.placeholder(tf.int32, [None, item_num, item_spare_num], name='item_spar')
            self.itm_dens_ph = tf.placeholder(tf.float32, [None, item_num, item_dense_num], name='item_dens')
            self.hist_spar_ph = tf.placeholder(tf.int32, [None, hist_num, hist_sapre_num], name='hist_spare')
            self.label_ph = tf.placeholder(tf.int32, [None, item_num], name='click_label')

            self.build_embedding(self.usr_spar_ph, self.itm_spar_ph, self.hist_spar_ph, args)

            self._state = tf.concat([
                tf.reshape(self.uid_batch_embedded, [-1, args.user_embedding_size]),
                self.usr_dens_ph,
                tf.reshape(self.mid_batch_embedded, [-1, item_num*args.item_embedding_size]),
                tf.reshape(self.cat_batch_embedded, [-1, item_num*args.cate_embedding_size]),
                tf.reshape(self.cat1_batch_embedded, [-1, item_num*args.cate1_embedding_size]),
                tf.reshape(self.shop_batch_embedded, [-1, item_num*args.shop_embedding_size]),
                tf.reshape(self.price_batch_embedded, [-1, item_num*args.price_embedding_size]),
                tf.reshape(self.itm_dens_ph, [-1, item_num*item_dense_num]),
                tf.reshape(self.mid_his_batch_embedded, [-1, hist_num*args.item_embedding_size]),
                tf.reshape(self.cat_his_batch_embedded, [-1, hist_num*args.cate_embedding_size]),
                tf.reshape(self.cat1_his_batch_embedded, [-1, hist_num*args.cate1_embedding_size]),
                tf.reshape(self.shop_his_batch_embedded, [-1, hist_num*args.shop_embedding_size]),
                tf.reshape(self.price_his_batch_embedded, [-1, hist_num*args.price_embedding_size]),
            ], axis=-1)

            if args.full_sequence:
                self._item_embed = tf.concat([
                    tf.reshape(self.mid_batch_embedded, [-1, item_num, args.item_embedding_size]),
                    tf.reshape(self.cat_batch_embedded, [-1, item_num, args.cate_embedding_size]),
                    tf.reshape(self.cat1_batch_embedded, [-1, item_num, args.cate1_embedding_size]),
                    tf.reshape(self.shop_batch_embedded, [-1, item_num, args.shop_embedding_size]),
                    tf.reshape(self.price_batch_embedded, [-1, item_num, args.price_embedding_size]),
                    tf.reshape(self.itm_dens_ph, [-1, item_num, item_dense_num])], axis=-1)

            self._action_probs = get_dnn(self._state, [512, 256, 64, max_action], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.softmax], "actor")
            
            log_loss = tf.losses.log_loss(self._action_probs, self.label_ph)
            # mse_loss = tf.losses.mean_squared_error(self._action_choices, self.label_ph)
            # train
            self._loss = log_loss
            # eval
            self._eval_loss = log_loss

            self._optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
            self._train_op = self._optimizer.minimize(self._loss)
    
    def get_state_embed(self, sess, user_spare, user_dense, item_spare, item_dense, hist_spare):
        return sess.run(self._state, feed_dict={
            self.usr_spar_ph: user_spare,
            self.usr_dens_ph: user_dense,
            self.itm_spar_ph: item_spare,
            self.itm_dens_ph: item_dense,
            self.hist_spar_ph: hist_spare 
        })
    
    def get_state_embed_item_embed(self, sess, user_spare, user_dense, item_spare, item_dense, hist_spare):
        return sess.run([self._state, self._item_embed], feed_dict={
            self.usr_spar_ph: user_spare,
            self.usr_dens_ph: user_dense,
            self.itm_spar_ph: item_spare,
            self.itm_dens_ph: item_dense,
            self.hist_spar_ph: hist_spare 
        })

    def predict(self, sess, user_spare, user_dense, item_spare, item_dense, hist_spare):
        return sess.run(self._action_probs, feed_dict={
            self.usr_spar_ph: user_spare,
            self.usr_dens_ph: user_dense,
            self.itm_spar_ph: item_spare,
            self.itm_dens_ph: item_dense,
            self.hist_spar_ph: hist_spare 
        })
    
    def evaluate(self, sess, user_spare, user_dense, item_spare, item_dense, hist_spare, label):
        loss, prob, label = sess.run([self._loss, self._action_probs, self.label_ph], feed_dict={
            self.usr_spar_ph: user_spare,
            self.usr_dens_ph: user_dense,
            self.itm_spar_ph: item_spare,
            self.itm_dens_ph: item_dense,
            self.hist_spar_ph: hist_spare,
            self.label_ph: label
        })
        return loss, prob, label

    def update(self, sess, user_spare, user_dense, item_spare, item_dense, hist_spare, label):
        _, loss = sess.run([self._train_op, self._loss], feed_dict={
            self.usr_spar_ph: user_spare,
            self.usr_dens_ph: user_dense,
            self.itm_spar_ph: item_spare,
            self.itm_dens_ph: item_dense,
            self.hist_spar_ph: hist_spare,
            self.label_ph: label
        })
        # print("log loss:", loss)
        
        return _, loss

    def build_embedding(self, user_spare_feature, item_spare_feature, hist_spare_feature, args=None):
        used_user_feature = {"user_id": 0}
        used_item_feature = {"item_id": 0, "cat_id": 1, "cat1_id": 2, "shop_id": 3, "price_id": 4}
        used_hist_feature = {"item_id": 0, "cat_id": 1, "cat1_id": 2, "shop_id": 3, "price_id": 4}
        # 创建Embedding
        self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [args.user_size, args.user_embedding_size])
        self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, user_spare_feature[:, 0])

        self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [args.item_size, args.item_embedding_size])
        self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, item_spare_feature[:, :, 0])
        self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, hist_spare_feature[:, :, 0])

        self.cat_embeddings_var = tf.get_variable("cat_embedding_var", [args.cate_size, args.cate_embedding_size])
        self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, item_spare_feature[:, :, 1])
        self.cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, hist_spare_feature[:, :, 1])

        self.cat1_embeddings_var = tf.get_variable("cat1_embedding_var", [args.cate1_size, args.cate1_embedding_size])
        self.cat1_batch_embedded = tf.nn.embedding_lookup(self.cat1_embeddings_var, item_spare_feature[:, :, 2])
        self.cat1_his_batch_embedded = tf.nn.embedding_lookup(self.cat1_embeddings_var, hist_spare_feature[:, :, 2])

        self.shop_embeddings_var = tf.get_variable("shop_embedding_var", [args.shop_size, args.shop_embedding_size])
        self.shop_batch_embedded = tf.nn.embedding_lookup(self.shop_embeddings_var, item_spare_feature[:, :, 3])
        self.shop_his_batch_embedded = tf.nn.embedding_lookup(self.shop_embeddings_var, hist_spare_feature[:, :, 3])

        self.price_embeddings_var = tf.get_variable("price_embedding_var", [args.price_size, args.price_embedding_size])
        self.price_batch_embedded = tf.nn.embedding_lookup(self.price_embeddings_var, item_spare_feature[:, :, 4])
        self.price_his_batch_embedded = tf.nn.embedding_lookup(self.price_embeddings_var, hist_spare_feature[:, :, 4])


class SLAvitoSessionActorOneHot:
    """The Supversived Actor"""

    def __init__(self, user_spare_num=4, user_dense_num=4, item_num=12, item_spare_num=2, item_dense_num=2,
                 hist_num=24, hist_sapre_num=2, num_actions=12, max_action=12, name="SL_CBU_Session", args=None):
        self._name = name
        self.state_dim = args.user_embedding_size * user_spare_num + (item_num * item_spare_num + hist_num * hist_sapre_num) * (
                    args.item_embedding_size) + item_num * item_dense_num

        with tf.variable_scope(self._name):
            self.usr_spar_ph = tf.placeholder(tf.int32, [None, user_spare_num], name='user_spar')
            self.usr_dens_ph = tf.placeholder(tf.float32, [None, user_dense_num], name='user_dense')
            self.itm_spar_ph = tf.placeholder(tf.int32, [None, item_num, item_spare_num], name='item_spar')
            self.itm_dens_ph = tf.placeholder(tf.float32, [None, item_num, item_dense_num], name='item_dens')
            self.hist_spar_ph = tf.placeholder(tf.int32, [None, hist_num, hist_sapre_num], name='hist_spare')
            self.label_ph = tf.placeholder(tf.int32, [None, item_num], name='click_label')
            self.is_train = tf.placeholder(tf.bool, [], name='is_train')

            self.build_embedding(self.usr_spar_ph, self.itm_spar_ph, self.hist_spar_ph, args)

            self._state = tf.concat([
                tf.reshape(self.uid_batch_embedded, [-1, args.user_embedding_size * user_spare_num]),
                tf.reshape(self.mid_batch_embedded, [-1, item_num * args.item_embedding_size * item_spare_num]),
                tf.reshape(self.itm_dens_ph, [-1, item_num * item_dense_num]),
                tf.reshape(self.mid_his_batch_embedded, [-1, hist_num * args.item_embedding_size * hist_sapre_num]),
            ], axis=-1)

            if args.full_sequence:
                self._item_embed = tf.concat([
                    tf.reshape(self.mid_batch_embedded, [-1, item_num, args.item_embedding_size * item_spare_num]),
                    tf.reshape(self.itm_dens_ph, [-1, item_num, item_dense_num])], axis=-1)

            # self._action_probs = get_dnn(self._state, [512, 256, 64, max_action], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.softmax], "actor")
            self._action_probs = self.build_mlp_net(self._state, max_action, scope="actor")

            log_loss = tf.losses.log_loss(self.label_ph, self._action_probs)
            # mse_loss = tf.losses.mean_squared_error(self._action_choices, self.label_ph)
            # train
            self._loss = log_loss
            # eval
            self._eval_loss = log_loss

            self._optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
            self._train_op = self._optimizer.minimize(self._loss)

    def build_mlp_net(self, inp, max_action, keep_prob=0.7, layer=(512, 256, 64), scope='mlp'):
        with tf.variable_scope(scope):
            inp = tf.layers.batch_normalization(inputs=inp, name='mlp_bn', training=self.is_train)
            for i, hidden_num in enumerate(layer):
                fc = tf.layers.dense(inp, hidden_num, activation=tf.nn.relu, name='fc' + str(i))
                inp = tf.nn.dropout(fc, keep_prob, name='dp' + str(i))
            final = tf.layers.dense(inp, max_action, activation=None, name='fc_final')
            y_pred = tf.nn.softmax(final)
            y_pred = tf.reshape(y_pred, [-1, max_action])
        return y_pred

    def get_state_embed(self, sess, user_spare, user_dense, item_spare, item_dense, hist_spare):
        return sess.run(self._state, feed_dict={
            self.usr_spar_ph: user_spare,
            self.usr_dens_ph: user_dense,
            self.itm_spar_ph: item_spare,
            self.itm_dens_ph: item_dense,
            self.hist_spar_ph: hist_spare
        })

    def get_state_embed_item_embed(self, sess, user_spare, user_dense, item_spare, item_dense, hist_spare):
        return sess.run([self._state, self._item_embed], feed_dict={
            self.usr_spar_ph: user_spare,
            self.usr_dens_ph: user_dense,
            self.itm_spar_ph: item_spare,
            self.itm_dens_ph: item_dense,
            self.hist_spar_ph: hist_spare
        })

    def predict(self, sess, user_spare, user_dense, item_spare, item_dense, hist_spare):
        return sess.run(self._action_probs, feed_dict={
            self.usr_spar_ph: user_spare,
            self.usr_dens_ph: user_dense,
            self.itm_spar_ph: item_spare,
            self.itm_dens_ph: item_dense,
            self.hist_spar_ph: hist_spare,
            self.is_train: False
        })

    def evaluate(self, sess, user_spare, user_dense, item_spare, item_dense, hist_spare, label):
        loss, prob, label = sess.run([self._loss, self._action_probs, self.label_ph], feed_dict={
            self.usr_spar_ph: user_spare,
            self.usr_dens_ph: user_dense,
            self.itm_spar_ph: item_spare,
            self.itm_dens_ph: item_dense,
            self.hist_spar_ph: hist_spare,
            self.label_ph: label,
            self.is_train: False
        })
        return loss, prob, label

    def update(self, sess, user_spare, user_dense, item_spare, item_dense, hist_spare, label):
        _, loss = sess.run([self._train_op, self._loss], feed_dict={
            self.usr_spar_ph: user_spare,
            self.usr_dens_ph: user_dense,
            self.itm_spar_ph: item_spare,
            self.itm_dens_ph: item_dense,
            self.hist_spar_ph: hist_spare,
            self.label_ph: label,
            self.is_train: True
        })
        # print("log loss:", loss)

        return _, loss

    def build_embedding(self, user_spare_feature, item_spare_feature, hist_spare_feature, args=None):
        used_user_feature = {"user_id": 0}
        used_item_feature = {"item_id": 0, "cat_id": 1, "cat1_id": 2, "shop_id": 3, "price_id": 4}
        used_hist_feature = {"item_id": 0, "cat_id": 1, "cat1_id": 2, "shop_id": 3, "price_id": 4}
        # 创建Embedding
        self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [args.user_size, args.user_embedding_size], initializer=tf.truncated_normal_initializer)
        self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, user_spare_feature)

        self.mid_embeddings_var = tf.get_variable("item_embedding_var", [args.item_size, args.item_embedding_size], initializer=tf.truncated_normal_initializer)
        self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, item_spare_feature)
        self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, hist_spare_feature)
        # self.mid_batch_embedded = tf.gather(self.mid_embeddings_var, item_spare_feature)
        # self.mid_his_batch_embedded = tf.gather(self.mid_embeddings_var, hist_spare_feature)



class SLCbuActor:
    """The Supversived Actor"""
    def __init__(self, user_spare_num=8, user_dense_num=14, item_num=12, item_spare_num=37, item_dense_num=29, hist_num=62, hist_sapre_num=7, num_actions=1, max_action=12, name="SL_CBU", args=None):
        self._name = name
        self.state_dim = args.user_embedding_size + user_dense_num + (item_num+hist_num) * (args.item_embedding_size + args.cate_embedding_size + args.cate1_embedding_size + \
                                                                                            args.shop_embedding_size + args.price_embedding_size) + item_num * item_dense_num
        with tf.variable_scope(self._name):
            self.usr_spar_ph = tf.placeholder(tf.int32, [None, user_spare_num], name='user_spar')
            self.usr_dens_ph = tf.placeholder(tf.float32, [None, user_dense_num], name='user_dense')
            self.itm_spar_ph = tf.placeholder(tf.int32, [None, item_num, item_spare_num], name='item_spar')
            self.itm_dens_ph = tf.placeholder(tf.float32, [None, item_num, item_dense_num], name='item_dens')
            self.hist_spar_ph = tf.placeholder(tf.int32, [None, hist_num, hist_sapre_num], name='hist_spare')
            self.reward_ph = tf.placeholder(tf.float32, [None, 2], name='reward_click_conversion')

            self.build_embedding(self.usr_spar_ph, self.itm_spar_ph, self.hist_spar_ph, args)

            self._state = tf.concat([
                tf.reshape(self.uid_batch_embedded, [-1, args.user_embedding_size]),
                self.usr_dens_ph,
                tf.reshape(self.mid_batch_embedded, [-1, item_num*args.item_embedding_size]),
                tf.reshape(self.cat_batch_embedded, [-1, item_num*args.cate_embedding_size]),
                tf.reshape(self.cat1_batch_embedded, [-1, item_num*args.cate1_embedding_size]),
                tf.reshape(self.shop_batch_embedded, [-1, item_num*args.shop_embedding_size]),
                tf.reshape(self.price_batch_embedded, [-1, item_num*args.price_embedding_size]),
                tf.reshape(self.itm_dens_ph, [-1, item_num*item_dense_num]),
                tf.reshape(self.mid_his_batch_embedded, [-1, hist_num*args.item_embedding_size]),
                tf.reshape(self.cat_his_batch_embedded, [-1, hist_num*args.cate_embedding_size]),
                tf.reshape(self.cat1_his_batch_embedded, [-1, hist_num*args.cate1_embedding_size]),
                tf.reshape(self.shop_his_batch_embedded, [-1, hist_num*args.shop_embedding_size]),
                tf.reshape(self.price_his_batch_embedded, [-1, hist_num*args.price_embedding_size]),
            ], axis=-1)


            # self._state = tf.placeholder(dtype=tf.float32, shape=[None, state_dim], name='state')
            self._action = tf.placeholder(dtype=tf.int32, shape=[None, num_actions], name='true_action')
            self._var = tf.placeholder(dtype=tf.float32, shape=[], name="various") 
            _action_probs = get_dnn(self._state, [1024, 512, 256, 64, num_actions], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.sigmoid], "actor")
            self._action_probs = _action_probs * max_action

            constant = tf.cast(self._var, dtype=tf.float32)
            self._vars = tf.fill([tf.shape(self._state)[0], 1], constant)
            self.dist = tf.distributions.Normal(self._action_probs, self._vars)
            self._float_actions = tf.cast(self._action, dtype=tf.float32)
            self.log_prob = tf.reduce_sum(self.dist.log_prob(self._float_actions), axis=-1)
            
            
            # action_loss = tf.losses.mean_squared_error(self._action_probs, self._action)
            # 使用 reward loss 加权 loss
            reward_loss = tf.cast(tf.reduce_sum(self.reward_ph, axis=-1) * 10, tf.float32)
            true_action = tf.cast(self._action, tf.float32)
            action_loss = tf.cast(tf.pow(self._action_probs - true_action, 2), tf.float32)
            action_loss = tf.reduce_sum(tf.multiply(action_loss, reward_loss))
            self._loss = action_loss

            self._optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
            self._train_op = self._optimizer.minimize(self._loss)

    def get_state_embed(self, sess, user_spare, user_dense, item_spare, item_dense, hist_spare):
        return sess.run(self._state, feed_dict={
            self.usr_spar_ph: user_spare,
            self.usr_dens_ph: user_dense,
            self.itm_spar_ph: item_spare,
            self.itm_dens_ph: item_dense,
            self.hist_spar_ph: hist_spare 
        })

    def predict(self, sess, user_spare, user_dense, item_spare, item_dense, hist_spare):
        return sess.run(self._action_probs, feed_dict={
            self.usr_spar_ph: user_spare,
            self.usr_dens_ph: user_dense,
            self.itm_spar_ph: item_spare,
            self.itm_dens_ph: item_dense,
            self.hist_spar_ph: hist_spare 
        })
    
    def cal_log(self, sess, user_spare, user_dense, item_spare, item_dense, hist_spare, a, var):
        return sess.run(self.log_prob, feed_dict={
            self.usr_spar_ph: user_spare,
            self.usr_dens_ph: user_dense,
            self.itm_spar_ph: item_spare,
            self.itm_dens_ph: item_dense,
            self.hist_spar_ph: hist_spare,
            self._action: a, 
            self._var: var
        })
    
    def evaluate(self, sess, user_spare, user_dense, item_spare, item_dense, hist_spare, a, rewards):
        return sess.run(self._loss, feed_dict={
            self.usr_spar_ph: user_spare,
            self.usr_dens_ph: user_dense,
            self.itm_spar_ph: item_spare,
            self.itm_dens_ph: item_dense,
            self.hist_spar_ph: hist_spare,
            self._action: a,
            self.reward_ph: rewards
        })

    def update(self, sess, user_spare, user_dense, item_spare, item_dense, hist_spare, a, rewards):
        _, loss = sess.run([self._train_op, self._loss], feed_dict={
            self.usr_spar_ph: user_spare,
            self.usr_dens_ph: user_dense,
            self.itm_spar_ph: item_spare,
            self.itm_dens_ph: item_dense,
            self.hist_spar_ph: hist_spare,
            self._action: a,
            self.reward_ph: rewards
        })
        return _, loss

    def build_embedding(self, user_spare_feature, item_spare_feature, hist_spare_feature, args=None):
        used_user_feature = {"user_id": 0}
        used_item_feature = {"item_id": 0, "cat_id": 1, "cat1_id": 2, "shop_id": 3, "price_id": 4}
        used_hist_feature = {"item_id": 0, "cat_id": 1, "cat1_id": 2, "shop_id": 3, "price_id": 4}
        # 创建Embedding
        self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [args.user_size, args.user_embedding_size])
        self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, user_spare_feature[:, 0])

        self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [args.item_size, args.item_embedding_size])
        self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, item_spare_feature[:, :, 0])
        self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, hist_spare_feature[:, :, 0])

        self.cat_embeddings_var = tf.get_variable("cat_embedding_var", [args.cate_size, args.cate_embedding_size])
        self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, item_spare_feature[:, :, 1])
        self.cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, hist_spare_feature[:, :, 1])

        self.cat1_embeddings_var = tf.get_variable("cat1_embedding_var", [args.cate1_size, args.cate1_embedding_size])
        self.cat1_batch_embedded = tf.nn.embedding_lookup(self.cat1_embeddings_var, item_spare_feature[:, :, 2])
        self.cat1_his_batch_embedded = tf.nn.embedding_lookup(self.cat1_embeddings_var, hist_spare_feature[:, :, 2])

        self.shop_embeddings_var = tf.get_variable("shop_embedding_var", [args.shop_size, args.shop_embedding_size])
        self.shop_batch_embedded = tf.nn.embedding_lookup(self.shop_embeddings_var, item_spare_feature[:, :, 3])
        self.shop_his_batch_embedded = tf.nn.embedding_lookup(self.shop_embeddings_var, hist_spare_feature[:, :, 3])

        self.price_embeddings_var = tf.get_variable("price_embedding_var", [args.price_size, args.price_embedding_size])
        self.price_batch_embedded = tf.nn.embedding_lookup(self.price_embeddings_var, item_spare_feature[:, :, 4])
        self.price_his_batch_embedded = tf.nn.embedding_lookup(self.price_embeddings_var, hist_spare_feature[:, :, 4])

class SLCbuActorOneHot:
    """The Supversived Actor"""
    def __init__(self, user_spare_num=8, user_dense_num=14, item_num=12, item_spare_num=37, item_dense_num=29, hist_num=62, hist_sapre_num=7, num_actions=1, max_action=12, name="SL_CBU", args=None):
        self._name = name
        self.state_dim = args.user_embedding_size + user_dense_num + (item_num+hist_num) * (args.item_embedding_size + args.cate_embedding_size + args.cate1_embedding_size + \
                                                                                            args.shop_embedding_size + args.price_embedding_size) + item_num * item_dense_num

        with tf.variable_scope(self._name):
            self.usr_spar_ph = tf.placeholder(tf.int32, [None, user_spare_num], name='user_spar')
            self.usr_dens_ph = tf.placeholder(tf.float32, [None, user_dense_num], name='user_dense')
            self.itm_spar_ph = tf.placeholder(tf.int32, [None, item_num, item_spare_num], name='item_spar')
            self.itm_dens_ph = tf.placeholder(tf.float32, [None, item_num, item_dense_num], name='item_dens')
            self.hist_spar_ph = tf.placeholder(tf.int32, [None, hist_num, hist_sapre_num], name='hist_spare')
            self.reward_ph = tf.placeholder(tf.float32, [None, 2], name='reward_click_conversion')
            self.target_distribution = tf.constant([1/max_action] * max_action, dtype=tf.float32)
            self.target_distribution = tf.tile(tf.expand_dims(self.target_distribution, 0), [tf.shape(self.usr_spar_ph)[0], 1])

            self.build_embedding(self.usr_spar_ph, self.itm_spar_ph, self.hist_spar_ph, args)

            self._state = tf.concat([
                tf.reshape(self.uid_batch_embedded, [-1, args.user_embedding_size]),
                self.usr_dens_ph,
                tf.reshape(self.mid_batch_embedded, [-1, item_num*args.item_embedding_size]),
                tf.reshape(self.cat_batch_embedded, [-1, item_num*args.cate_embedding_size]),
                tf.reshape(self.cat1_batch_embedded, [-1, item_num*args.cate1_embedding_size]),
                tf.reshape(self.shop_batch_embedded, [-1, item_num*args.shop_embedding_size]),
                tf.reshape(self.price_batch_embedded, [-1, item_num*args.price_embedding_size]),
                tf.reshape(self.itm_dens_ph, [-1, item_num*item_dense_num]),
                tf.reshape(self.mid_his_batch_embedded, [-1, hist_num*args.item_embedding_size]),
                tf.reshape(self.cat_his_batch_embedded, [-1, hist_num*args.cate_embedding_size]),
                tf.reshape(self.cat1_his_batch_embedded, [-1, hist_num*args.cate1_embedding_size]),
                tf.reshape(self.shop_his_batch_embedded, [-1, hist_num*args.shop_embedding_size]),
                tf.reshape(self.price_his_batch_embedded, [-1, hist_num*args.price_embedding_size]),
            ], axis=-1)

            self._action = tf.placeholder(dtype=tf.int32, shape=[None, num_actions], name='true_action')
            self._action_probs = get_dnn(self._state, [512, 256, 64, max_action], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.softmax], "actor")
            self._var = tf.placeholder(dtype=tf.float32, shape=[], name="various")
            self._action_choices = tf.argmax(self._action_probs, axis=-1)
            self._action_choices = tf.reshape(self._action_choices, [tf.shape(self._state)[0], 1])


            constant = tf.cast(self._var, dtype=tf.float32)
            self._vars = tf.fill([tf.shape(self._state)[0], 1], constant)
            action_choice = tf.cast(self._action_choices, tf.float32)
            self.dist = tf.distributions.Normal(action_choice, self._vars)
            self._float_actions = tf.cast(self._action, dtype=tf.float32)
            self.log_prob = tf.reduce_sum(self.dist.log_prob(self._float_actions), axis=-1)

            reward_loss = tf.cast(tf.reduce_sum(self.reward_ph, axis=-1), tf.float32)
            
            # loss
            # log_loss = tf.reduce_mean(tf.multiply(tf.nn.softmax_cross_entropy_with_logits(labels=self._action, logits=self._action_probs), reward_loss))
            # reward > 0
            self.big_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self._action, logits=self._action_probs)
            # light_loss  = -(self._action_probs * tf.log(self._action_probs))
            self.kl_divergence = tf.reduce_sum(self.target_distribution * tf.log(self.target_distribution / self._action_probs), axis=-1)
            print("big loss shape:", tf.shape(self.big_loss))
            print("kl shape:", tf.shape(self.kl_divergence))
            log_loss = tf.reduce_mean(tf.where(tf.greater(reward_loss, 0), self.big_loss, self.kl_divergence))
            
            # log_loss = tf.losses.log_loss(self._action_probs, self._action)
            mse_loss = tf.losses.mean_squared_error(self._action_choices, self._action)
            # train
            self._loss = log_loss
            # eval
            self._eval_loss = mse_loss

            self._optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
            self._train_op = self._optimizer.minimize(self._loss)

    def predict(self, sess, user_spare, user_dense, item_spare, item_dense, hist_spare):
        return sess.run(self._action_probs, feed_dict={
            self.usr_spar_ph: user_spare,
            self.usr_dens_ph: user_dense,
            self.itm_spar_ph: item_spare,
            self.itm_dens_ph: item_dense,
            self.hist_spar_ph: hist_spare 
        })
    
    def cal_log(self, sess, user_spare, user_dense, item_spare, item_dense, hist_spare, a, var):
        return sess.run(self.log_prob, feed_dict={
            self.usr_spar_ph: user_spare,
            self.usr_dens_ph: user_dense,
            self.itm_spar_ph: item_spare,
            self.itm_dens_ph: item_dense,
            self.hist_spar_ph: hist_spare,
            self._action: a, 
            self._var: var
        })
    
    def evaluate(self, sess, user_spare, user_dense, item_spare, item_dense, hist_spare, a, reward):
        return sess.run(self._loss, feed_dict={
            self.usr_spar_ph: user_spare,
            self.usr_dens_ph: user_dense,
            self.itm_spar_ph: item_spare,
            self.itm_dens_ph: item_dense,
            self.hist_spar_ph: hist_spare,
            self._action: a,
            self.reward_ph: reward
        })

    def update(self, sess, user_spare, user_dense, item_spare, item_dense, hist_spare, a, reward):
        _, loss, big_loss, kl_div, true_action, prob_action, distrubution = sess.run([self._train_op, self._loss, self.big_loss, self.kl_divergence, self._action, self._action_probs, self.target_distribution], feed_dict={
            self.usr_spar_ph: user_spare,
            self.usr_dens_ph: user_dense,
            self.itm_spar_ph: item_spare,
            self.itm_dens_ph: item_dense,
            self.hist_spar_ph: hist_spare,
            self._action: a,
            self.reward_ph: reward
        })
        print("log loss:", loss)
        print("big loss:", big_loss)
        print("kl div:", kl_div)
        print("true action:", true_action)
        print("prob action:", prob_action)
        print("distribution:", distrubution)
        
        return _, loss

    def build_embedding(self, user_spare_feature, item_spare_feature, hist_spare_feature, args=None):
        used_user_feature = {"user_id": 0}
        used_item_feature = {"item_id": 0, "cat_id": 1, "cat1_id": 2, "shop_id": 3, "price_id": 4}
        used_hist_feature = {"item_id": 0, "cat_id": 1, "cat1_id": 2, "shop_id": 3, "price_id": 4}
        # 创建Embedding
        self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [args.user_size, args.user_embedding_size])
        self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, user_spare_feature[:, 0])

        self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [args.item_size, args.item_embedding_size])
        self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, item_spare_feature[:, :, 0])
        self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, hist_spare_feature[:, :, 0])

        self.cat_embeddings_var = tf.get_variable("cat_embedding_var", [args.cate_size, args.cate_embedding_size])
        self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, item_spare_feature[:, :, 1])
        self.cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, hist_spare_feature[:, :, 1])

        self.cat1_embeddings_var = tf.get_variable("cat1_embedding_var", [args.cate1_size, args.cate1_embedding_size])
        self.cat1_batch_embedded = tf.nn.embedding_lookup(self.cat1_embeddings_var, item_spare_feature[:, :, 2])
        self.cat1_his_batch_embedded = tf.nn.embedding_lookup(self.cat1_embeddings_var, hist_spare_feature[:, :, 2])

        self.shop_embeddings_var = tf.get_variable("shop_embedding_var", [args.shop_size, args.shop_embedding_size])
        self.shop_batch_embedded = tf.nn.embedding_lookup(self.shop_embeddings_var, item_spare_feature[:, :, 3])
        self.shop_his_batch_embedded = tf.nn.embedding_lookup(self.shop_embeddings_var, hist_spare_feature[:, :, 3])

        self.price_embeddings_var = tf.get_variable("price_embedding_var", [args.price_size, args.price_embedding_size])
        self.price_batch_embedded = tf.nn.embedding_lookup(self.price_embeddings_var, item_spare_feature[:, :, 4])
        self.price_his_batch_embedded = tf.nn.embedding_lookup(self.price_embeddings_var, hist_spare_feature[:, :, 4])

class SLActor:
    """The Supversived Actor"""
    def __init__(self, state_dim, num_actions, max_action, name, args):
        self._name = name

        with tf.variable_scope(self._name):
            self._state = tf.placeholder(dtype=tf.float32, shape=[None, state_dim], name='state')
            self._action = tf.placeholder(dtype=tf.int32, shape=[None, num_actions], name='true_action')
            self._var = tf.placeholder(dtype=tf.float32, shape=[], name="various") 
            _action_probs = get_dnn(self._state, [512, 256, 64, num_actions], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.sigmoid], "actor")
            self._action_probs = _action_probs * max_action

            constant = tf.cast(self._var, dtype=tf.float32)
            self._vars = tf.fill([tf.shape(self._state)[0], 1], constant)
            self.dist = tf.distributions.Normal(self._action_probs, self._vars)
            self._float_actions = tf.cast(self._action, dtype=tf.float32)
            self.log_prob = tf.reduce_sum(self.dist.log_prob(self._float_actions), axis=-1)
            

            action_loss = tf.losses.mean_squared_error(self._action_probs, self._action)
            self._loss = action_loss

            self._optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
            self._train_op = self._optimizer.minimize(self._loss)

    def predict(self, sess, s):
        return sess.run(self._action_probs, {self._state: s})
    
    def cal_log(self, sess, s, a, var):
        return sess.run(self.log_prob, {self._state: s, self._action: a, self._var: var})
    
    def evaluate(self, sess, s, a):
        return sess.run(self._loss, {self._state: s, self._action: a})

    def update(self, sess, s, a):
        _, loss = sess.run([self._train_op, self._loss], {self._state: s, self._action: a,})
        return _, loss

    
class SLActorOneHot:
    """The Supversived Actor"""
    def __init__(self, state_dim, num_actions, max_action, name, args):
        self._name = name

        with tf.variable_scope(self._name):
            self._state = tf.placeholder(dtype=tf.float32, shape=[None, state_dim], name='state')
            self._action = tf.placeholder(dtype=tf.int32, shape=[None, num_actions], name='true_action')
            self._action_probs = get_dnn(self._state, [512, 256, 64, max_action], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.softmax], "actor")
            self._var = tf.placeholder(dtype=tf.float32, shape=[], name="various")
            self._action_choices = tf.argmax(self._action_probs, axis=-1)
            self._action_choices = tf.reshape(self._action_choices, [tf.shape(self._state)[0], 1])


            constant = tf.cast(self._var, dtype=tf.float32)
            self._vars = tf.fill([tf.shape(self._state)[0], 1], constant)
            action_choice = tf.cast(self._action_choices, tf.float32)
            self.dist = tf.distributions.Normal(action_choice, self._vars)
            self._float_actions = tf.cast(self._action, dtype=tf.float32)
            self.log_prob = tf.reduce_sum(self.dist.log_prob(self._float_actions), axis=-1)
            
            # loss
            log_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._action, logits=self._action_probs))
            # log_loss = tf.losses.log_loss(self._action_probs, self._action)
            mse_loss = tf.losses.mean_squared_error(self._action_choices, self._action)
            # train
            self._loss = log_loss
            # eval
            self._eval_loss = mse_loss

            self._optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
            self._train_op = self._optimizer.minimize(self._loss)

    def predict(self, sess, s):
        return sess.run(self._action_choices, {self._state: s})
    
    def cal_log(self, sess, s, a, var):
        return sess.run(self.log_prob, {self._state: s, self._action: a, self._var: var})
    
    def evaluate(self, sess, s, a):
        return sess.run(self._eval_loss, {self._state: s, self._action: a})

    def update(self, sess, s, a):
        _, loss = sess.run([self._train_op, self._loss], {self._state: s, self._action: a,})
        return _, loss

class Actor:
    """The actor class"""

    def __init__(self, sess, state_dim, num_actions, max_action, name, args):
        self._sess = sess
        self._name = name

        with tf.variable_scope(self._name):
            self._state = tf.placeholder(dtype=tf.float32, shape=[None, state_dim], name='state')
            self._action = tf.placeholder(dtype=tf.int32, shape=[None, num_actions], name='true_action')
            _action_probs = get_dnn(self._state, [32, 32, num_actions], [tf.nn.relu, tf.nn.relu, tf.nn.sigmoid])
            self._action_probs = _action_probs * max_action
            

            action_loss = tf.losses.mean_squared_error(self._action_probs, self._action)
            self._loss = action_loss

            self._optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
            self._train_op = self._optimizer.minimize(self._loss)

    def predict(self, s):
        return self._sess.run(self._action_probs, {self._state: s})
    
    def evaluate(self, s, a):
        return self._sess.run(self._loss, {self._state: s, self._action: a})

    def update(self, s, a):
        _, loss = self._sess.run([self._train_op, self._loss], {self._state: s, self._action: a,})
        return _, loss

class SL(object):
    def __init__(self, sess, args):
        self.sess = sess
        tf.reset_default_graph()
        self.graph = tf.Graph() 
        state_dim, action_dim, max_action=args.state_dim, args.action_dim, args.max_action
        self.actor = Actor(self.sess, state_dim, action_dim, max_action, "actor", args)
    
    def select_action(self, state):
        return self.actor.predict(state)
    
    def train(self, batch):
        states = batch["states"]
        actions = batch["actions"]

        _, loss = self.actor.update(states, actions)
        return loss
    
    def eval(self, batch):
        states = batch["states"]
        actions = batch["actions"]
        
        loss = self.actor.evaluate(states, actions)
        
        return loss
    
    def save(self, filename):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, save_path=filename)
            print('Save model:', filename)

    def load(self, filename):
        with self.graph.as_default():
            ckpt = tf.train.get_checkpoint_state(filename)
            if ckpt and ckpt.model_checkpoint_path:
                saver = tf.train.Saver()
                saver.restore(sess=self.sess, save_path=ckpt.model_checkpoint_path)
                print('Restore model:', ckpt.model_checkpoint_path)