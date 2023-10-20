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
from sl import SLActor, SLActorOneHot


class Actor:
    """The actor class"""

    def __init__(self, state_dim, num_actions, max_action, multi_num, name, args):
        self._name = name

        with tf.variable_scope(self._name):
            self._state = tf.placeholder(dtype=tf.float32, shape=[None, state_dim], name='state')
            self._action = tf.placeholder(dtype=tf.int32, shape=[None, num_actions], name='true_action')
            self._ref_action = tf.placeholder(dtype=tf.int32, shape=[None, num_actions], name='ref_action')
            self._qvalue = tf.placeholder(dtype=tf.float32, shape=[None, multi_num], name='critic_value')
            self._weight = tf.placeholder(dtype=tf.float32, shape=[None, multi_num], name='critic_loss_weight')
            self._ratio = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="ratio")
            self._var = tf.placeholder(dtype=tf.float32, shape=[], name="various")
            
            constant = tf.cast(self._var, dtype=tf.float32)
            self._vars = tf.fill([tf.shape(self._state)[0], 1], constant)
            
            _action_probs = get_dnn(self._state, [512, 256, 64, num_actions], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.sigmoid], "actor")
            self._action_probs = _action_probs * max_action
            
            self.dist = tf.distributions.Normal(self._action_probs, self._vars)
            self._float_actions = tf.cast(self._action, dtype=tf.float32)
            self.log_prob = tf.reduce_sum(self.dist.log_prob(self._float_actions), axis=-1)
            

            action_loss = tf.losses.mean_squared_error(self._action_probs, self._action)
            qvalue_wight = tf.multiply(self._qvalue, self._weight)
            critic_loss = -tf.reduce_mean(tf.reduce_mean(qvalue_wight, axis=-1), axis=-1)

            # one stage loss
            self._loss = critic_loss + action_loss * args.bc_loss_coeff

            
            # two stage loss
            tmp_loss = tf.reduce_mean(qvalue_wight, axis=-1)
            ratio_tmp_loss = -tf.reduce_mean(tf.multiply(self._ratio, tmp_loss), axis=-1)
            ref_loss = tf.losses.mean_squared_error(self._action_probs, self._ref_action)
            self._two_stage_loss = args.awac_loss_coef * ratio_tmp_loss + args.bc_loss_coeff * action_loss + args.kl_loss_coeff * ref_loss

            self._optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
            self._train_op = self._optimizer.minimize(self._loss)

            self._two_train_op = self._optimizer.minimize(self._two_stage_loss)


    def predict(self, sess, s):
        return sess.run(self._action_probs, {self._state: s})
    
    def cal_log(self, sess, s, a, var):
        return sess.run(self.log_prob, {self._state: s, self._action: a, self._var: var})

    def update(self, sess, s, a, qvalue, weight):
        _, loss = sess.run([self._train_op, self._loss], {self._state: s, self._action: a, self._qvalue: qvalue, self._weight: weight})
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
        self.multi_num = multi_num
        self._build_model()

    def _build_model(self):
        with tf.variable_scope(self._name):
            self._state = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim], name='state')
            self._action = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim], name='action')
            self._target = tf.placeholder(dtype=tf.float32, shape=[None, self.multi_num], name='target')

            q0 = get_dnn_critic(self._state, self._action, [512, 256, 64], [tf.nn.relu, tf.nn.relu, tf.nn.relu], [64, 32, 1], [tf.nn.relu, tf.nn.relu, None], "critic_dnn_00")
            q1 = get_dnn_critic(self._state, self._action, [512, 256, 64], [tf.nn.relu, tf.nn.relu, tf.nn.relu], [64, 32, 1], [tf.nn.relu, tf.nn.relu, None], "critic_dnn_01")
            q2 = get_dnn_critic(self._state, self._action, [512, 256, 64], [tf.nn.relu, tf.nn.relu, tf.nn.relu], [64, 32, 1], [tf.nn.relu, tf.nn.relu, None], "critic_dnn_02")
            q3 = get_dnn_critic(self._state, self._action, [512, 256, 64], [tf.nn.relu, tf.nn.relu, tf.nn.relu], [64, 32, 1], [tf.nn.relu, tf.nn.relu, None], "critic_dnn_03")
            q4 = get_dnn_critic(self._state, self._action, [512, 256, 64], [tf.nn.relu, tf.nn.relu, tf.nn.relu], [64, 32, 1], [tf.nn.relu, tf.nn.relu, None], "critic_dnn_04")
            q5 = get_dnn_critic(self._state, self._action, [512, 256, 64], [tf.nn.relu, tf.nn.relu, tf.nn.relu], [64, 32, 1], [tf.nn.relu, tf.nn.relu, None], "critic_dnn_05")
            q6 = get_dnn_critic(self._state, self._action, [512, 256, 64], [tf.nn.relu, tf.nn.relu, tf.nn.relu], [64, 32, 1], [tf.nn.relu, tf.nn.relu, None], "critic_dnn_06")
            # q7 = get_dnn_critic(self._state, self._action, [32], [tf.nn.relu], [32, 1], [tf.nn.relu, None], "critic_dnn_07")
            
            self._out = tf.concat([q0, q1, q2, q3, q4, q5, q6], axis=-1)

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
        state_dim, action_dim, max_action, multi_num, discount, tau=args.state_dim, args.action_dim, args.max_action, args.multi_num, args.discount, args.tau
        self.discount = discount
        self.args = args
        self.tau = tau
        self.weight = [args.reward_click, args.reward_like,args.reward_follow, args.reward_comment, args.reward_forward, args.reward_hate, args.reward_play_time]
        # reset graph
        tf.reset_default_graph()
        self.graph = tf.Graph() 
        with tf.variable_scope("one_stage"):
            self.actor = Actor(state_dim, action_dim, max_action, multi_num, "current_actor", args)
            self.target_actor = Actor(state_dim, action_dim, max_action, multi_num, "target_actor", args)
            self.critic = Critic(state_dim, action_dim, multi_num, "current_critic", args)
            self.target_critic = Critic(state_dim, action_dim, multi_num, "target_critic", args)

        # two stage model
        with tf.variable_scope("two_stage"):
            self.TS_actor = Actor(state_dim, action_dim, max_action, multi_num, "current_actor", args)
            self.TS_target_actor = Actor(state_dim, action_dim, max_action, multi_num, "target_actor", args)
            self.TS_critic = Critic(state_dim, action_dim, multi_num, "current_critic", args)
            self.TS_target_critic = Critic(state_dim, action_dim, multi_num, "target_critic", args)

        if args.behavior_onehot:
            self.sl_actor = SLActorOneHot(state_dim, action_dim, max_action, "sl_actor_onehot", args)
        else:
            self.sl_actor = SLActor(state_dim, action_dim, max_action, "sl_actor", args)
        self._build_ratio(args)
        self._build_soft_update()
        self._build_two_stage_update()

    def _build_ratio(self, args):
        constrained_dist = tf.distributions.Normal(self.actor._action_probs, self.actor._vars)
        current_dist = tf.distributions.Normal(self.TS_actor._action_probs, self.TS_actor._vars)
        if args.new_ratio:
            self.ratio = tf.clip_by_value(tf.exp(constrained_dist.log_prob(self.actor._action_probs)), clip_value_min=0, clip_value_max=1)
        else:
            self.ratio = tf.clip_by_value(tf.exp(constrained_dist.log_prob(self.actor._action) - current_dist.log_prob(self.TS_actor._action)), clip_value_min=0, clip_value_max=1)
  
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

    def _build_two_stage_update(self):
        self.two_update_actor_target_op = []
        self.two_update_critic_target_op = []

        actor_var_list = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if ("current_actor" in v.name and "two_stage" in v.name)]
        target_actor_var_list = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if ("target_actor" in v.name and "two_stage" in v.name)]

        critic_var_list = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if ("current_critic" in v.name and "two_stage" in v.name)]
        target_critic_var_list = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if ("target_critic" in v.name and "two_stage" in v.name)]


        for main_var, target_var in zip(actor_var_list, target_actor_var_list):
            update_target_var = target_var.assign(self.tau * main_var + (1 - self.tau) * target_var)
            self.two_update_actor_target_op.append(update_target_var)

        for main_var, target_var in zip(critic_var_list, target_critic_var_list):
            update_target_var = target_var.assign(self.tau * main_var + (1 - self.tau) * target_var)
            self.two_update_critic_target_op.append(update_target_var)

    def select_action(self, sess, state):
        return self.actor.predict(sess, state)
    
    def TS_select_action(self, sess, state):
        return self.TS_actor.predict(sess, state)
    
    def sl_select_action(self, sess, state):
        return self.sl_actor.predict(sess, state)
    
    def cal_ratio(self, sess, states, actions):
        log_prob_policy = self.actor.cal_log(sess, states, actions, 25)
        log_prob_sl_policy = self.sl_actor.cal_log(sess, states, actions, 25)
        ratios = np.clip(np.exp(log_prob_policy - log_prob_sl_policy), 0, 10)
        return ratios
    
    def cal_two_stage_ratio(self, sess, state, actions):
        ratios = sess.run(self.ratio, {self.actor._state: state,
                                       self.TS_actor._state: state,
                                       self.actor._action: actions,
                                       self.TS_actor._action: actions,
                                       self.actor._var: self.args.sigma,
                                       self.TS_actor._var: self.args.sigma})
        return ratios
    
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

    def two_stage_save(self, sess, filename):
        var_list = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if ("two_stage" in v.name)]
        saver = tf.train.Saver(var_list=var_list)
        saver.save(sess, save_path=filename)
        print('Save Two Stage model:', filename)
    
    def two_stage_load(self, sess, filename):
        ckpt = tf.train.get_checkpoint_state(filename)
        var_list = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if ("two_stage" in v.name)]
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver(var_list=var_list)
            saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
            print('Restore model:', ckpt.model_checkpoint_path)
    
    def sl_eval(self, sess, batch):
        states = batch["states"]
        actions = batch["actions"]
        loss = self.sl_actor.evaluate(sess, states, actions)
        return loss
    
    def sl_save(self, sess, filename):
        # print("Trainable var:", tf.trainable_variables())
        if self.args.behavior_onehot:
            var_list = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if "sl_actor_onehot" in v.name]
        else:
            var_list = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if ("sl_actor" in v.name and "onehot" not in v.name)]
        saver = tf.train.Saver(var_list=var_list)
        saver.save(sess, save_path=filename)
        print('Save model:', filename)

    def sl_load(self, sess, filename):
        ckpt = tf.train.get_checkpoint_state(filename)
        if self.args.behavior_onehot:
            var_list = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if "sl_actor_onehot" in v.name]
        else:
            var_list = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if ("sl_actor" in v.name and "onehot" not in v.name)]
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver(var_list=var_list)
            saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
            print('Restore model:', ckpt.model_checkpoint_path)

    def soft_update(self, sess):
        sess.run([self.update_actor_target_op, self.update_critic_target_op])

    def soft_update_two_stage(self, sess):
        sess.run([self.two_update_actor_target_op, self.two_update_critic_target_op])

    def sl_train(self, sess, batch):
        states = batch["states"]
        actions = batch["actions"]
        # print("state:", states)
        # print("action:", actions)
        _, loss = self.sl_actor.update(sess, states, actions)
        return loss
    
    def train(self, sess, batch):
        states = batch["states"]
        next_states = batch["next_states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]
        weights = [self.weight for _ in range(len(states))]


        target_q = self.target_critic.predict(sess, next_states, self.target_actor.predict(sess, next_states))
        target_q = rewards + self.discount * dones * target_q
        # 更新 critic
        critic_loss = self.critic.update(sess, states, actions, target_q)

        # 更新 actor
        qvalue = self.critic.predict(sess, states, self.actor.predict(sess, states))
        actor_loss = self.actor.update(sess, states, actions, qvalue, weights)

        return actor_loss, critic_loss
    
    def two_stage_train(self, sess, batch):
        states = batch["states"]
        next_states = batch["next_states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]
        weights = [self.weight for _ in range(len(states))]

        target_q = self.TS_target_critic.predict(sess, next_states, self.TS_target_actor.predict(sess, next_states))
        # print("First Target Q:", target_q)
        # print("Discount :", self.discount)
        # print("Dones :", dones)
        # print("Rewards :", rewards)
        target_q = rewards + self.discount * dones * target_q
        # print("Second Taeget Q:", target_q)

        # 更新 critic
        critic_loss = self.TS_critic.update(sess, states, actions, target_q)


        ref_actions = self.actor.predict(sess, states)
        print("Ref Action: ", ref_actions)
        ratio = self.cal_two_stage_ratio(sess, states, actions)
        print("Ratio: ",ratio)
        qvalue = self.TS_critic.predict(sess, states, self.actor.predict(sess, states))
        print("Qvalue: ",qvalue)

        two_stage_actor_loss = self.TS_actor.two_stage_update(sess, states, actions, ratio, ref_actions, qvalue, weights)




        return two_stage_actor_loss, critic_loss




