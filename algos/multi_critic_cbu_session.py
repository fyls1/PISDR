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
from sl import SLCbuSessionActorOneHot
import os


class Actor:
    """The actor class"""

    def __init__(self, state_dim, num_actions, max_action, multi_num, name, args):
        self._name = name
        self.args = args

        with tf.variable_scope(self._name):
            # from SL model embedding
            self._state = tf.placeholder(dtype=tf.float32, shape=[None, state_dim], name='state')
            self._action = tf.placeholder(dtype=tf.int32, shape=[None, num_actions], name='true_action')
            self._ref_action = tf.placeholder(dtype=tf.int32, shape=[None, num_actions], name='ref_action')
            self._qvalue = tf.placeholder(dtype=tf.float32, shape=[None, multi_num], name='critic_value')
            self._weight = tf.placeholder(dtype=tf.float32, shape=[None, multi_num], name='critic_loss_weight')
            self._ratio = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="ratio")
            self._var = tf.placeholder(dtype=tf.float32, shape=[], name="various")
            self.reward_ph = tf.placeholder(tf.float32, [None, 3], name='reward_click_conversion_score')
            
            constant = tf.cast(self._var, dtype=tf.float32)
            self._vars = tf.fill([tf.shape(self._state)[0], 1], constant)
            
            _action_probs = get_dnn(self._state, [512, 256, 64, max_action], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.softmax], "actor")
            self._action_probs = _action_probs
            self._action_choices = tf.argmax(self._action_probs, axis=-1)
            self._action_choices = tf.reshape(self._action_choices, [tf.shape(self._state)[0], 1])
            
            self.dist = tf.distributions.Normal(self._action_probs, self._vars)
            self._float_actions = tf.cast(self._action, dtype=tf.float32)
            self.log_prob = tf.reduce_sum(self.dist.log_prob(self._float_actions), axis=-1)
            

            reward_loss = tf.cast(tf.reduce_sum(self.reward_ph, axis=-1) * 10, tf.float32)

            # action_loss = tf.reduce_mean(tf.multiply(tf.nn.softmax_cross_entropy_with_logits(labels=self._action, logits=self._action_probs), reward_loss))
            action_loss = tf.losses.log_loss(labels=self._action, predictions=self._action_probs)
            qvalue_wight = tf.multiply(self._qvalue, self._weight)
            critic_loss = -tf.reduce_mean(tf.reduce_mean(qvalue_wight, axis=-1), axis=-1)

            # one stage loss
            self._loss = critic_loss + action_loss * args.bc_loss_coeff

            
            # two stage loss
            tmp_loss = tf.reduce_mean(qvalue_wight, axis=-1)
            ratio_tmp_loss = -tf.reduce_mean(tf.multiply(self._ratio, tmp_loss), axis=-1)
            if args.ac_onehot:
                ref_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._ref_action, logits=self._action_probs))
            else:
                ref_loss = tf.losses.mean_squared_error(self._action_probs, self._ref_action)
            self._two_stage_loss = args.awac_loss_coef * ratio_tmp_loss + args.bc_loss_coeff * action_loss + args.kl_loss_coeff * ref_loss

            self._optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
            self._train_op = self._optimizer.minimize(self._loss)

            self._two_train_op = self._optimizer.minimize(self._two_stage_loss)


    def predict(self, sess, s):
        return sess.run(self._action_probs, {self._state: s})
    
    def cal_log(self, sess, s, a, var):
        return sess.run(self.log_prob, {self._state: s, self._action: a, self._var: var})

    def update(self, sess, s, a, qvalue, weight, reward):
        _, loss = sess.run([self._train_op, self._loss], {self._state: s, self._action: a, self._qvalue: qvalue, self._weight: weight, self.reward_ph: reward})
        return loss
    
    def two_stage_update(self, sess, s, a, ratio, ref_a, qvalue, weight):
        _, two_stage_loss = sess.run([self._two_train_op, self._two_stage_loss], 
                               {self._state: s, 
                                self._action: a, 
                                self._qvalue: qvalue, 
                                self._weight: weight,
                                self._ratio: ratio,
                                self._ref_action: ref_a})
        return two_stage_loss

class Critic:
    """The critic class"""

    def __init__(self, state_dim, action_dim, multi_num, name, args):
        self.args = args
        self._name = name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = args.max_action
        self.multi_num = multi_num
        self._build_model()

    def _build_model(self):
        with tf.variable_scope(self._name):
            self._state = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim], name='state')
            self._action = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim], name='action')
            self._target = tf.placeholder(dtype=tf.float32, shape=[None, self.multi_num], name='target')

            q0 = get_dnn_critic(self._state, self._action, [512, 256, 64], [tf.nn.relu, tf.nn.relu, tf.nn.relu], [64, 32, 1], [tf.nn.relu, tf.nn.relu, None], "critic_dnn_00") # ctr
            q1 = get_dnn_critic(self._state, self._action, [512, 256, 64], [tf.nn.relu, tf.nn.relu, tf.nn.relu], [64, 32, 1], [tf.nn.relu, tf.nn.relu, None], "critic_dnn_01") # conver
            q2 = get_dnn_critic(self._state, self._action, [512, 256, 64], [tf.nn.relu, tf.nn.relu, tf.nn.relu], [64, 32, 1], [tf.nn.relu, tf.nn.relu, None], "critic_dnn_02") # score
            # q3 = get_dnn_critic(self._state, self._action, [512, 256, 64], [tf.nn.relu, tf.nn.relu, tf.nn.relu], [64, 32, 1], [tf.nn.relu, tf.nn.relu, None], "critic_dnn_03")
            # q4 = get_dnn_critic(self._state, self._action, [512, 256, 64], [tf.nn.relu, tf.nn.relu, tf.nn.relu], [64, 32, 1], [tf.nn.relu, tf.nn.relu, None], "critic_dnn_04")
            # q5 = get_dnn_critic(self._state, self._action, [512, 256, 64], [tf.nn.relu, tf.nn.relu, tf.nn.relu], [64, 32, 1], [tf.nn.relu, tf.nn.relu, None], "critic_dnn_05")
            # q6 = get_dnn_critic(self._state, self._action, [512, 256, 64], [tf.nn.relu, tf.nn.relu, tf.nn.relu], [64, 32, 1], [tf.nn.relu, tf.nn.relu, None], "critic_dnn_06")
            # q7 = get_dnn_critic(self._state, self._action, [32], [tf.nn.relu], [32, 1], [tf.nn.relu, None], "critic_dnn_07")
            
            self._out = tf.concat([q0, q1, q2], axis=-1)

            self._loss = tf.losses.mean_squared_error(self._out, self._target)
            self._optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
            self._update_step = self._optimizer.minimize(self._loss)

    def predict(self, sess, s, a):
        return sess.run(self._out, feed_dict={self._state: s, self._action: a})

    def update(self, sess, s, a, target):
        _, loss = sess.run([self._update_step, self._loss], feed_dict={self._state: s, self._action: a,self._target: target})
        return loss


class MULTI_CRITIC(object):
    def __init__(self, args):
        # state dim from SL model embedding total sum
        state_dim, action_dim, max_action, multi_num, discount, tau = args.state_dim, args.action_dim, args.max_action, args.multi_num, args.discount, args.tau
        self.discount = discount
        self.args = args
        self.tau = tau
        # self.weight = [args.reward_click, args.reward_like,args.reward_follow, args.reward_comment, args.reward_forward, args.reward_hate, args.reward_play_time]
        self.weight = [args.reward_click, args.reward_conservation, args.reward_score]
        # reset graph
        tf.reset_default_graph()
        self.graph = tf.Graph() 

        self.sl_actor = SLCbuSessionActorOneHot(name="sl_actor_onehot", args=args)

        if self.sl_actor:
            state_dim = self.sl_actor.state_dim


        with tf.variable_scope("one_stage"):
            self.actor = Actor(state_dim, action_dim, max_action, multi_num, "current_actor", args)
            self.target_actor = Actor(state_dim, action_dim, max_action, multi_num, "target_actor", args)
            if args.ac_onehot:
                self.critic = Critic(state_dim, max_action, multi_num, "current_critic", args)
                self.target_critic = Critic(state_dim, max_action, multi_num, "target_critic", args)
            else:
                self.critic = Critic(state_dim, action_dim, multi_num, "current_critic", args)
                self.target_critic = Critic(state_dim, action_dim, multi_num, "target_critic", args)

        self._build_soft_update()

    def _build_soft_update(self):
        self.update_actor_target_op = []
        self.update_critic_target_op = []

        actor_var_list = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if ("current_actor" in v.name and "one_stage" in v.name)]
        target_actor_var_list = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if ("target_actor" in v.name and "one_stage" in v.name)]

        critic_var_list = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if ("current_critic" in v.name and "one_stage" in v.name)]
        target_critic_var_list = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if ("target_critic" in v.name and "one_stage" in v.name)]


        for main_var, target_var in zip(actor_var_list, target_actor_var_list):
            update_target_var = target_var.assign(self.tau * main_var + (1 - self.tau) * target_var)
            self.update_actor_target_op.append(update_target_var)

        for main_var, target_var in zip(critic_var_list, target_critic_var_list):
            update_target_var = target_var.assign(self.tau * main_var + (1 - self.tau) * target_var)
            self.update_critic_target_op.append(update_target_var)

    def select_action(self, sess, batch):
        # Todo: use cbu feature to SL get state embedding
        states = batch["states"]
        user_spare = states["user_spare_feature"]
        user_dense = states["user_dense_feature"]
        item_spare = states["item_spare_feature"]
        item_dense = states["item_dense_feature"]
        hist_spare = states["hist_spare_feature"]
        input_state = self.sl_actor.get_state_embed(sess, user_spare, user_dense, item_spare, item_dense, hist_spare)
        return self.actor.predict(sess, input_state)
    
    def select_action_critic(self, sess, batch):
        # Todo: use cbu feature to SL get state embedding
        states = batch["states"]
        user_spare = states["user_spare_feature"]
        user_dense = states["user_dense_feature"]
        item_spare = states["item_spare_feature"]
        item_dense = states["item_dense_feature"]
        hist_spare = states["hist_spare_feature"]
        input_state = self.sl_actor.get_state_embed(sess, user_spare, user_dense, item_spare, item_dense, hist_spare)
        return self.actor.predict(sess, input_state), self.critic.predict(sess, input_state, self.actor.predict(sess, input_state)) 
    
    def sl_select_action(self, sess, batch):
        states = batch["states"]
        user_spare = states["user_spare_feature"]
        user_dense = states["user_dense_feature"]
        item_spare = states["item_spare_feature"]
        item_dense = states["item_dense_feature"]
        hist_spare = states["hist_spare_feature"]
        return self.sl_actor.predict(sess, user_spare, user_dense, item_spare, item_dense, hist_spare)
    
    def cal_ratio(self, sess, batch, actions):
        states = batch["states"]
        user_spare = states["user_spare_feature"]
        user_dense = states["user_dense_feature"]
        item_spare = states["item_spare_feature"]
        item_dense = states["item_dense_feature"]
        hist_spare = states["hist_spare"]
        input_state = self.sl_actor.get_state_embed(sess, user_spare, user_dense, item_spare, item_dense, hist_spare)
        log_prob_policy = self.actor.cal_log(sess, input_state, actions, 5)
        log_prob_sl_policy = self.sl_actor.cal_log(sess, user_spare, user_dense, item_spare, item_dense, hist_spare, actions, 5)
        ratios = np.clip(np.exp(log_prob_policy - log_prob_sl_policy), 0, 10)
        return ratios
    
    def total_save(self, sess, ckpt_path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=ckpt_path)
        print("save total model:", ckpt_path)
    
    def total_load(self, sess, ckpt_dir):
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(ckpt_dir)
        print("load total model:", ckpt)
        saver.restore(sess, ckpt)
    
    def sl_ac_load(self, sess, sl_ckpt_dir, ac_ckpt_dir):
        print("Loading SL && AC Model...")
        sl_var_list = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if "sl_actor_onehot" in v.name]
        ac_var_list = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if ("one_stage" in v.name)]
        sl_saver = tf.train.Saver(var_list=sl_var_list)
        ac_saver = tf.train.Saver(var_list=ac_var_list)
        sl_ckpt = tf.train.latest_checkpoint(sl_ckpt_dir)
        ac_ckpt = tf.train.latest_checkpoint(ac_ckpt_dir)
        print("load sl model:", sl_ckpt)
        sl_saver.restore(sess, sl_ckpt)
        print("load ac model:", ac_ckpt)
        ac_saver.restore(sess, ac_ckpt)
        print("Load Success!")

    def one_stage_save_emb(self, sess, filename):
        var_list = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if ("one_stage" in v.name and "embedding" in v.name)]
        saver = tf.train.Saver(var_list=var_list)
        saver.save(sess, save_path=filename)
        print('Save One Stage Embedding model:', filename)
    
    def one_stage_save(self, sess, filename):
        var_list = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if ("one_stage" in v.name)]
        saver = tf.train.Saver(var_list=var_list)
        saver.save(sess, save_path=filename)
        print('Save One Stage model:', filename)
    
    def one_stage_load(self, sess, filename):
        ckpt = tf.train.get_checkpoint_state(filename)
        var_list = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if ("one_stage" in v.name)]
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver(var_list=var_list)
            saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
            print('Restore model:', ckpt.model_checkpoint_path)
    
    def sl_eval(self, sess, batch):
        user_spare = batch["user_spare_feature"]
        user_dense = batch["user_dense_feature"]
        item_spare = batch["item_spare_feature"]
        item_dense = batch["item_dense_feature"]
        hist_spare = batch["hist_spare_feature"]
        labels = batch["ctr_label"]
        loss, prob, label = self.sl_actor.evaluate(sess, user_spare, user_dense, item_spare, item_dense, hist_spare, labels)
        return loss, prob, label

    def sl_eval_rl(self, sess, batch):
        states = batch["states"]
        user_spare = states["user_spare_feature"]
        user_dense = states["user_dense_feature"]
        item_spare = states["item_spare_feature"]
        item_dense = states["item_dense_feature"]
        hist_spare = states["hist_spare_feature"]
        labels = batch["actions"]["ctr_label"]
        loss, prob, label = self.sl_actor.evaluate(sess, user_spare, user_dense, item_spare, item_dense, hist_spare, labels)
        return loss, prob, label
    
    def sl_save(self, sess, filename):
        # print("Trainable var:", tf.trainable_variables())

        var_list = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if "sl_actor_onehot" in v.name]
        saver = tf.train.Saver(var_list=var_list)
        saver.save(sess, save_path=filename)
        print('Save model:', filename)

    def sl_load(self, sess, filename):
        ckpt = tf.train.get_checkpoint_state(filename)
        var_list = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if "sl_actor_onehot" in v.name]
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver(var_list=var_list)
            saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
            print('Restore model:', ckpt.model_checkpoint_path)

    def soft_update(self, sess):
        sess.run([self.update_actor_target_op, self.update_critic_target_op])

    def sl_train(self, sess, batch):
        # print("batch info:", batch)
        # ["user_spare_feature", "user_dense_feature", "item_spare_feature", "item_dense_feature", "dynamic_item_spare_feature", "dynamic_item_dense_feature", "hist_spare_feature"]

        user_spare = batch["user_spare_feature"]
        user_dense = batch["user_dense_feature"]
        item_spare = batch["item_spare_feature"]
        item_dense = batch["item_dense_feature"]
        hist_spare = batch["hist_spare_feature"]
        labels = batch["ctr_label"]

        # print("user shape:", user_spare.shape)
        # print("user dense:", user_dense.shape)
        # print("item spare:", item_spare.shape)
        # print("item dense:", item_dense.shape)
        # print("hist spare:", hist_spare.shape)

        # print("state:", states)
        # print("action:", actions)
        _, loss = self.sl_actor.update(sess, user_spare, user_dense, item_spare, item_dense, hist_spare, labels)
        return loss
    
    def sl_critic_pre(self, sess, batch):
        states = batch["states"]

        user_spare = states["user_spare_feature"]
        user_dense = states["user_dense_feature"]
        item_spare = states["item_spare_feature"]
        item_dense = states["item_dense_feature"]
        hist_spare = states["hist_spare_feature"]

        input_state = self.sl_actor.get_state_embed(sess, user_spare, user_dense, item_spare, item_dense, hist_spare)
        qvalue = self.critic.predict(sess, input_state, self.sl_actor.predict(sess, user_spare, user_dense, item_spare, item_dense, hist_spare))
        return qvalue

    
    def train(self, sess, batch):
        states = batch["states"]
        next_states = batch["next_states"]
        actions = batch["actions"]["ctr_label"]
        rewards = batch["rewards"]
        dones = batch["dones"]["done"]
        

        # if self.args.ac_onehot:
        #     true_actions = actions
        #     actions = np.eye(self.args.max_action)[actions].reshape([len(actions), self.args.max_action])

        user_spare = states["user_spare_feature"]
        user_dense = states["user_dense_feature"]
        item_spare = states["item_spare_feature"]
        item_dense = states["item_dense_feature"]
        hist_spare = states["hist_spare_feature"]

        next_user_spare = next_states["user_spare_feature"]
        next_user_dense = next_states["user_dense_feature"]
        next_item_spare = next_states["item_spare_feature"]
        next_item_dense = next_states["item_dense_feature"]
        next_hist_spare = next_states["hist_spare_feature"]

        batch_size = len(user_spare)

        weights = [self.weight for _ in range(batch_size)]

        # print("weights:", weights)

        ctr_rewards = np.reshape(rewards["ctr_rewards"], [batch_size, 1])
        conver_rewards = np.reshape(rewards["conver_rewards"], [batch_size, 1])
        score_rewards = np.reshape(rewards["score_rewards"], [batch_size, 1])

        # print("ctr reward:", ctr_rewards)
        # print("conver reward:", conver_rewards)
        # print("score reward:", score_rewards)

        total_reward = np.concatenate([ctr_rewards, conver_rewards, score_rewards], axis=-1)


        # print("user spare:", user_spare)
        # print("total reward:", total_reward)
        # print("dones:", dones)
        
        input_state = self.sl_actor.get_state_embed(sess, user_spare, user_dense, item_spare, item_dense, hist_spare)

        next_input_state = self.sl_actor.get_state_embed(sess, next_user_spare, next_user_dense, next_item_spare, next_item_dense, next_hist_spare)


        target_q = self.target_critic.predict(sess, next_input_state, self.target_actor.predict(sess, next_input_state))
        target_q = total_reward + self.discount * dones * target_q
        # 更新 critic
        critic_loss = self.critic.update(sess, input_state, actions, target_q)

        # 更新 actor
        qvalue = self.critic.predict(sess, input_state, self.actor.predict(sess, input_state))
        actor_loss = self.actor.update(sess, input_state, actions, qvalue, weights, total_reward)

        return actor_loss, critic_loss




