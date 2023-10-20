#coding=utf-8
import os
import pickle
import sys
import os
import tensorflow as tf
import random
import time
import numpy as np
import datetime
import json
import argparse
from algos.mult_critic import MULTI_CRITIC
from read_kuaishou_from_odps_tf_io import DataInput, get_aggregated_batch4kuaishou
import datetime

def load_parse_from_json(parse, setting_path):
    with open(setting_path, 'r') as f:
        setting = json.load(f)
    parse_dict = vars(parse)
    for k, v in setting.items():
        parse_dict[k] = v
    return parse

def create_policy(args):
    if args.algo == "multi_critic":
        return MULTI_CRITIC(args)
    
def offline_ab(sess, agent, dataset, args):
    print("="*50, "offline_ab testing", "="*50)
    total_ratio = 0
    estimated_rewards = np.zeros(args.reward_dim)
    actions = []
    bc_actions = []
    start_time = time.time()
    for _, eval_datas in DataInput(dataset, args.batch_size):
        batch_datas = get_aggregated_batch4kuaishou(eval_datas)
        policy_action = agent.select_action(sess, batch_datas["states"])
        behavior_policy_action = agent.sl_select_action(sess, batch_datas["states"])
        actions.extend(policy_action)
        bc_actions.extend(behavior_policy_action)

        total_ratio += np.sum(agent.cal_ratio(sess, batch_datas["states"], batch_datas["actions"]))

    print("Finished calculating total_ratio", total_ratio, ". Time elapsed", time.time() - start_time)
    
    for _, eval_datas in DataInput(dataset, args.batch_size):
        batch_datas = get_aggregated_batch4kuaishou(eval_datas)
        policy_action = agent.select_action(sess, batch_datas["states"])
        behavior_policy_action = agent.sl_select_action(sess, batch_datas["states"])
        actions.extend(policy_action)
        bc_actions.extend(behavior_policy_action)

        ratios = agent.cal_ratio(sess, batch_datas["states"], batch_datas["actions"])
        ratios = ratios/total_ratio

        R = np.array(batch_datas["rewards"])
        ratios = np.reshape(ratios, [ratios.shape[0], 1])
        ratios = np.tile(ratios, [1, R.shape[1]])
        R = ratios * R

        estimated_rewards = estimated_rewards + np.sum(R, axis=0)

    print("Time elapsed:", time.time()-start_time)
    print("estimated_rewards:", estimated_rewards)

    return estimated_rewards


def run(train_dateset, test_dateset, args):
    date = datetime.datetime.now().strftime("%Y%m%d%H%M")
    model_name = '{}_{}'.format(date, args.algo)
    data_set_name = "kuaishou"
    log_path = os.path.join(buckets, 'logs/{}/{}'.format(data_set_name, date))
    model_path = os.path.join(buckets, 'save_model/{}/{}'.format(data_set_name, date))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    save_name = 'save_model_{}_{}_ckpt'.format(data_set_name, model_name)
    save_sl_name = 'save_sl_model_{}_{}_ckpt'.format(data_set_name, model_name)
    save_path = os.path.join(model_path, save_name)
    save_sl_path = os.path.join(model_path, save_sl_name)
    load_sl_path = os.path.join(buckets, args.load_sl_path)
    load_one_stage_path = os.path.join(buckets, args.load_one_stage_path)
    print("save model path", save_path)
    print("save sl model path", save_sl_path)

    print("save parase file...")
    save_parase_json_path = os.path.join(log_path, "params.json")
    with open(save_parase_json_path, 'w') as f:
        json.dump(vars(args), f)
    print("save parase file done!")
    train_process_json_path = os.path.join(log_path, "train_monitor.json")

    tf.reset_default_graph()
    agent = create_policy(args)

    # gpu settings
    gpu_options = tf.GPUOptions(allow_growth=True)

    training_monitor = {
        'actor_loss': [],
        'critic_loss': [],
        'max_rewards': [],
        'max_improve': []
    }
    
    # training process
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        start_time = time.time()

        #Todo: build SL model
        if args.train_sl:
            print("="*50,"Train SL model","="*50)
            for e in range(int(args.bc_steps)):
                train_losses = []
                for _, train_datas in DataInput(test_dateset, args.batch_size):
                    batch_datas = get_aggregated_batch4kuaishou(train_datas)
                    train_loss = agent.sl_train(sess, batch_datas)
                    train_losses.append(train_loss)

                if e % 20 == 0:
                    losses = []
                    for _, eval_datas in DataInput(test_dateset, args.batch_size):
                        eval_batch_datas = get_aggregated_batch4kuaishou(eval_datas)
                        loss = agent.sl_eval(sess, eval_batch_datas)
                        losses.append(loss)
                    print("EPOCH: ", e, "; TRAIN LOSS:", np.mean(train_losses), "; EVAL LOSS:", np.mean(losses))

                    if args.save_sl:
                        agent.sl_save(sess, save_sl_path)
            load_sl_path = save_sl_path

        print("Load SL Model...")
        agent.sl_load(sess, load_sl_path)

        if args.train_two_stage:
            print("Load one stage model ...", load_one_stage_path)
            agent.one_stage_load(sess, load_one_stage_path)

        ref_mean_reward = np.array([5.33783117e-01, 1.23087050e-02, 4.60753661e-04, 3.22527563e-03, 1.11897318e-03, 2.30376831e-04, 1.28546980e+00])
        max_rewards = [-10] * 7
        max_improvement = [-10] * 7

        for e in range(int(args.max_steps)):
            total_actor_loss = []
            total_critic_loss = []
            for _, train_datas in DataInput(train_dateset, args.batch_size):
                batch_datas = get_aggregated_batch4kuaishou(train_datas)
                if args.train_two_stage:
                    actor_loss, critic_loss = agent.two_stage_train(sess, batch_datas)
                    agent.soft_update_two_stage(sess)
                    training_monitor["actor_loss"].append(actor_loss)
                    training_monitor["critic_loss"].append(critic_loss)
                else:
                    actor_loss, critic_loss = agent.train(sess, batch_datas)
                    agent.soft_update(sess)
                    training_monitor["actor_loss"].append(actor_loss)
                    training_monitor["critic_loss"].append(critic_loss)
                total_actor_loss.append(actor_loss)
                total_critic_loss.append

            if e % args.save_every == 0:
                print('At ieration', e, 'time elapsed', time.time() - start_time,'mean actor loss:', np.mean(actor_loss), 'mean critic loss :', np.mean(critic_loss))
                estimated_rewards = offline_ab(sess, agent, test_dateset, args)
                print("Improvement:", (estimated_rewards-ref_mean_reward)/ref_mean_reward*100)
                if estimated_rewards[-1] > max_rewards[-1]:
                    max_rewards = estimated_rewards
                    max_improvement = (estimated_rewards-ref_mean_reward)/ref_mean_reward*100
                print("Max rewards: ", max_rewards)
                print("Max improvement: ", max_improvement)
                training_monitor["max_rewards"].append(max_rewards.tolist())
                training_monitor["max_improve"].append(max_improvement.tolist())

                if args.save_model:
                    if args.train_two_stage:
                        agent.two_stage_save(sess, os.path.join(model_path, "two_stage_"+save_name))
                    else:
                        agent.one_stage_save(sess, os.path.join(model_path, "one_stage_"+save_name))
                
                # with open(train_process_json_path, "w") as f:
                #     json.dump(training_monitor, f)
                

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default="test", type=str, help='exp name')
    parser.add_argument('--setting_path', type=str, default='./configs/multi_critic.json', help='setting dir')
    FLAGS, _ = parser.parse_known_args()
    return FLAGS


if __name__ == '__main__':
    # parameters
    random.seed(1234)
    args = parse_args()
    if args.setting_path:
        args = load_parse_from_json(args, args.setting_path)

    tf.app.flags.DEFINE_string("tables", "", "tables info")
    tf.app.flags.DEFINE_string("buckets", "", "buckets info")
    FLAGS = tf.app.flags.FLAGS
    print("tables:" + FLAGS.tables)
    # 0 train 1 small train 2 test 
    tables = FLAGS.tables
    tables = FLAGS.tables.split(",")
    print("split tables", tables)

    
    # FLAGS = tf.app.flags.FLAGS
    buckets = FLAGS.buckets
    print("buckets:", buckets)

    print("="*50, "load train dataset", "="*50)

    train_reader = tf.python_io.TableReader(tables[1])
    total_train_num = train_reader.get_row_count() 
    train_dataset_set = train_reader.read(total_train_num)
    train_reader.close()

    test_reader = tf.python_io.TableReader(tables[2])
    total_test_num = test_reader.get_row_count() 
    test_dataset_set = test_reader.read(total_test_num)
    test_reader.close()


    # ===== training =====

    run(train_dataset_set, test_dataset_set, args)