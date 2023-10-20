#coding=utf-8
import os
import pickle as pkl
import sys
import os
import tensorflow as tf
import random
import time
import numpy as np
import datetime
import json
import argparse
# from algos.multi_critic_cbu_session_seq import MULTI_CRITIC
from read_cbu_from_odps import *
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
        if args.is_sequence:
            from algos.multi_critic_cbu_session_seq import MULTI_CRITIC
        elif args.full_sequence:
            from algos.multi_critic_cbu_session_full_seq import MULTI_CRITIC
        else:
            from algos.multi_critic_cbu_session import MULTI_CRITIC
        return MULTI_CRITIC(args)
    
def constract_offline_input(offline_datas):
    '''
    from : ["user_spare", "user_dense", "item_spare", "item_dense", "hist_spare", "ctr_label", "conver_label"]
    to: ["states", "ctr_label", "mask"]
    "states" : user_spare_feature + user_dense_feature + item_spare_feature + item_dense_feature + hist_spare_feature(62:50+12)
    '''
    batch = len(offline_datas["user_spare"])
    buffers_used_keys_total_size = {
        "user_spare": 8, 
        "user_dense": 14, 
        "item_spare": 12*37, 
        "item_dense": 12*29, 
        "hist_spare": 50*7, 
        "ctr_label": 1, 
        "conver_label": 1
    }
    user_spare = offline_datas["user_spare"]
    user_dense = offline_datas["user_dense"]
    item_spare = offline_datas["item_spare"]
    item_dense = offline_datas["item_dense"]
    hist_spare = offline_datas["hist_spare"]
    ctr_label = offline_datas["ctr_label"]
    conver_label = offline_datas["conver_label"]
    hist_spare_added = np.zeros([batch, 12, 7])

    hist_total_spare = np.concatenate([hist_spare, hist_spare_added], axis=1)

    mask = np.ones([batch, 12])

    states = {
        "user_spare_feature": user_spare,
        "user_dense_feature": user_dense,
        "item_spare_feature": item_spare,
        "item_dense_feature": item_dense,
        "hist_spare": hist_total_spare
    }

    # states = np.concatenate(
    #     [
    #         user_spare,
    #         user_dense,
    #         item_spare,
    #         item_dense,
    #         hist_total_spare
    #     ], axis=-1
    # )

    return states, ctr_label, conver_label, mask

def update_state(states, actions, mask, times):
    '''
    1. update hist spare   取 item spare 的前5个 填充 hist 的 7 个
    2. update mask (done)
    '''
    for i in range(len(actions)):
        item_spare_single = states["item_spare_feature"][i, actions[i], :]
        states["hist_spare"][i, 50+actions[i], :] = item_spare_single[0:7]

    mask_points = [(i, actions[i]) for i in range(len(actions))]
    # print("mask points:",mask_points)
    # print("mask:", mask)
    for i, j in mask_points:
        mask[i, j] = 0


    return states, mask

def total_reward_batch(total_batch_preds, total_batch_labels, total_batch_dones):
    log_rewards = []
    log_times = []
    _time = 0
    sum_reward = 0
    for pred, label, done in zip(total_batch_preds, total_batch_labels, total_batch_dones):
        _time += 1
        _sum_reward = np.sum(np.array(pred) * np.array(label)).tolist()
        sum_reward += _sum_reward

        if done == 0:
            log_rewards.append(sum_reward)
            log_times.append(_time)
            sum_reward = 0
            _time = 0
            
    return sum(log_rewards), log_rewards, log_times

def evaluate_matrix(batch_preds, batch_labels):
    '''
    计算其他指标: MAP NDCG 
    '''
    batch_ndcg = []
    batch_map = []
    for pred, label in zip(batch_preds, batch_labels):
        ideal_dcg, dcg, AP_value, AP_count = 0, 0, 0, 0

        final = sorted(range(len(pred)), key=lambda k: pred[k], reverse=True)
        click = np.array(label)[final].tolist()  # reranked label
        gold = sorted(range(len(label)), key=lambda k: label[k], reverse=True)  # optimal list for ndcg

        for _i, _g in zip(range(1, 13), gold[:12]):
            dcg += (pow(2, click[_i - 1]) - 1) / (np.log2(_i + 1))
            ideal_dcg += (pow(2, label[_g]) - 1) / (np.log2(_i + 1))

            if click[_i - 1] >= 1:
                AP_count += 1
                AP_value += AP_count / _i

            _ndcg = float(dcg) / ideal_dcg if ideal_dcg != 0 else 0.5
            _map = float(AP_value) / AP_count if AP_count != 0 else 0.5
        
        batch_ndcg.append(_ndcg)
        batch_map.append(_map)
    return np.mean(np.array(batch_ndcg)), np.mean(np.array(batch_map))

def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """
    # print("array:", raw_arr)
    arr = sorted(raw_arr, key=lambda d:d[0], reverse=True)
    pos_rank_sum, pos = 0., 0.
    for i in range(len(arr)):
        record = arr[i]
        rank = len(arr) - i
        if record[1] == 1.:
            pos_rank_sum += rank
            pos += 1

    if pos == len(arr) or pos == 0:
        return 0.5

    auc = (pos_rank_sum - pos * (1 + pos) / 2) / (pos * (len(arr) - pos))

    return auc

def calc_auc_topk(raw_arr, k_scope):
    arr = sorted(raw_arr, key=lambda d:d[0], reverse=True)
    aucs = {}
    for k in k_scope:
        pos_rank_sum, pos = 0., 0.
        for i in range(min(k, len(arr))):
            record = arr[i]
            rank = len(arr) - i
            if record[1] == 1.:
                pos_rank_sum += rank
                pos += 1

        if pos == len(arr) or pos == 0:
            auc = 0.5
        else:
            auc = (pos_rank_sum - pos * (1 + pos) / 2) / (pos * (len(arr) - pos))
        aucs[k] = auc

    return aucs

def calculate_auc_metrix(prob, true_action):
    """
    prob: 模型重排好的序列
    true_action: click label
    """
    # print("prob:", prob)
    # print("true_action:", true_action)
    arr = true_action[prob] # 花式索引重新排序 action
    pos_rank_sum, pos = 0., 0.
    for i in range(len(arr)):
        record = arr[i]
        rank = len(arr) - i
        if record == 1.:
            pos_rank_sum += rank
            pos += 1

    if pos == len(arr) or pos == 0:
        return 0.5

    auc = (pos_rank_sum - pos * (1 + pos) / 2) / (pos * (len(arr) - pos))

    return auc

def evaluate_matrix_multi(batch_preds, batch_labels, scope_number=[3,5,10,12]):
    '''
    计算其他指标: MAP NDCG
    '''
    print("batch_preds", batch_preds)
    print("batch_labels", batch_labels)
    batch_ndcg = [[] for _ in range(len(scope_number))]
    batch_map = [[] for _ in range(len(scope_number))]
    for pred, label in zip(batch_preds, batch_labels):
        ideal_dcg, dcg, AP_value, AP_count = 0, 0, 0, 0

        final = sorted(range(len(pred)), key=lambda k: pred[k], reverse=True)
        click = np.array(label)[final].tolist()  # reranked label
        gold = sorted(range(len(label)), key=lambda k: label[k], reverse=True)  # optimal list for ndcg

        for i, scope in enumerate(scope_number):
            ideal_dcg, dcg, de_dcg, de_idcg, AP_value, AP_count, util = 0, 0, 0, 0, 0, 0, 0
            cur_scope = min(scope, len(label))
            for _i, _g in zip(range(1, cur_scope+1), gold[:cur_scope]):
                dcg += (pow(2, click[_i - 1]) - 1) / (np.log2(_i + 1))
                ideal_dcg += (pow(2, label[_g]) - 1) / (np.log2(_i + 1))

                if click[_i - 1] >= 1:
                    AP_count += 1
                    AP_value += AP_count / _i

            _ndcg = float(dcg) / ideal_dcg if ideal_dcg != 0 else 0.5
            _map = float(AP_value) / AP_count if AP_count != 0 else 0.5

            batch_ndcg[i].append(_ndcg)
            batch_map[i].append(_map)
    print("batch_map", np.array(batch_map).shape)
    print(batch_map)
    return np.mean(np.array(batch_ndcg), axis=-1), np.mean(np.array(batch_map), axis=-1)

def calculate_bacth_metrix(batch_prob, batch_label):
    res = []

    for prob, label in zip(batch_prob, batch_label):
        res.append(calculate_auc_metrix(prob, label))
    
    return np.mean(res)

def calculate_bacth_prob_metrix(batch_prob, batch_label):
    res = []

    for s_probs, s_labels in zip(batch_prob, batch_label):
        # print("prob:", s_probs)
        # print("label:", s_labels)
        arr = []

        init_score = 1
        for p, t in zip(s_probs, s_labels):
            if(p > 0):
                arr.append([p, t])
            init_score -= 0.05
        
    _auc = calc_auc(arr)

    res.append(_auc)
    return np.mean(res)

def calculate_bacth_prob_metrix_topk(batch_prob, batch_label, k_scope):
    res = {k : [] for k in k_scope}

    for s_probs, s_labels in zip(batch_prob, batch_label):
        arr = []

        for p, t in zip(s_probs, s_labels):
            if(p > 0):
                arr.append([p, t])
        
    _aucs = calc_auc_topk(arr, k_scope)

    for key in k_scope:
        res[key].append(_aucs[key])
    return res

def offline_topk(sess, agent, dataset, args, sl=False):
    print("="*50, "offline_tokp testing", "="*50)
    '''
    针对数据结构1的设计,将用户一次重排处理为12次的记录,因此需要重构输入输出,并在过程中持续更新状态
    根据session的信息 求离线的topk auc 等指标
    dataset: for get_aggregated_batch4cbu_offline
    '''
    k = [1, 3, 5, 10, 12]
    AUC_SEQ_MEAN = [[] for _ in range(len(k))]
    AUC_MEAN_MEAN = [[] for _ in range(len(k))]
    AUC_LAST_MEAN = [[] for _ in range(len(k))]
    for _, offline_datas in OfflineInput(dataset, args.offline_batch_size):
        batch_datas = get_aggregated_batch4cbu_offline(offline_datas)
        states, ctr_label, conver_label, mask = constract_offline_input(batch_datas)
        batch_datas = {
            "states": states
        }
        # 按顺序推荐12个商品
        seq_mask_batch_action = []
        last_select = []
        mean_select = np.zeros([len(mask), 12])
        for i in range(12):
            if sl:
                if args.behavior_onehot:
                    policy_action = agent.sl_select_action(sess, batch_datas)
                    seq_policy_action = np.argmax((policy_action * mask), axis=-1)
                    mean_select += policy_action
                    last_select = policy_action
                else:
                    policy_action = agent.sl_select_action(sess, batch_datas)
                    seq_policy_action = policy_action
            else:
                if args.ac_onehot:
                    policy_action = agent.select_action(sess, batch_datas)
                    seq_policy_action = np.argmax((policy_action * mask), axis=-1)
                    mean_select += policy_action
                    last_select = policy_action
                else:
                    policy_action = agent.select_action(sess, batch_datas)
                    seq_policy_action = policy_action

            seq_mask_batch_action.append(seq_policy_action)
            new_state, mask = update_state(states, seq_policy_action, mask, i)
            batch_datas["states"] = new_state


        batch_action = np.array(seq_mask_batch_action)
        batch_action = batch_action.T

        for index, _k in enumerate(k):
            _batch_action = batch_action[:, :_k]
            auc_seq_batch = calculate_bacth_metrix(_batch_action, ctr_label)
            AUC_SEQ_MEAN[index].append(auc_seq_batch)

            _mean_select = mean_select[:, :_k]
            _ctr_label = ctr_label[:, :_k]
            mean_auc_batch = calculate_bacth_prob_metrix(_mean_select, _ctr_label)
            # mean_ndcg, mean_map = evaluate_matrix(batch_preds=_mean_select, batch_labels=_ctr_label)
            AUC_MEAN_MEAN[index].append(mean_auc_batch)

            _last_select = last_select[:, :_k]
            last_auc_batch = calculate_bacth_prob_metrix(_last_select, _ctr_label)
            # mean_ndcg, mean_map = evaluate_matrix(batch_preds=_mean_select, batch_labels=_ctr_label)
            AUC_LAST_MEAN[index].append(last_auc_batch)
    for index, _k in enumerate(k):
        print("@{}: SEQ AUC  {}  MEAN AUC {}  LAST AUC  {}".format(_k, 
                                                                   np.mean(AUC_SEQ_MEAN[index]), 
                                                                   np.mean(AUC_MEAN_MEAN[index]), 
                                                                   np.mean(AUC_LAST_MEAN[index])))
        # print("SEQ AUC :", np.mean(AUC_SEQ_MEAN))
        # print("MEAN AUC:", np.mean(AUC_MEAN_MEAN))
        # print("LAST AUC:", np.mean(AUC_LAST_MEAN))
    return AUC_SEQ_MEAN, AUC_MEAN_MEAN, AUC_LAST_MEAN

def offline_ab(sess, agent, dataset, args):
    print("="*50, "offline_ab testing", "="*50)
    total_ratio = 0
    estimated_rewards = np.zeros(args.reward_dim)
    actions = []
    bc_actions = []
    start_time = time.time()
    for _, eval_datas in DataInput(dataset, args.batch_size):
        batch_datas = get_aggregated_batch4cbu_split(eval_datas)
        if args.ac_onehot:
            policy_action = np.argmax(agent.select_action(sess, batch_datas), axis=-1)
            behavior_policy_action = np.argmax(agent.sl_select_action(sess, batch_datas), axis=-1)
        else:
            policy_action = agent.select_action(sess, batch_datas)
            behavior_policy_action = agent.sl_select_action(sess, batch_datas)
        actions.extend(policy_action)
        bc_actions.extend(behavior_policy_action)

        total_ratio += np.sum(agent.cal_ratio(sess, batch_datas, batch_datas["actions"]))

    print("Finished calculating total_ratio", total_ratio, ". Time elapsed", time.time() - start_time)
    
    for _, eval_datas in DataInput(dataset, args.batch_size):
        batch_datas = get_aggregated_batch4cbu_split(eval_datas)
        if args.ac_onehot:
            policy_action = np.argmax(agent.select_action(sess, batch_datas), axis=-1)
            behavior_policy_action = np.argmax(agent.sl_select_action(sess, batch_datas), axis=-1)
        else:
            policy_action = agent.select_action(sess, batch_datas)
            behavior_policy_action = agent.sl_select_action(sess, batch_datas)
        actions.extend(policy_action)
        bc_actions.extend(behavior_policy_action)

        ratios = agent.cal_ratio(sess, batch_datas, batch_datas["actions"])
        ratios = ratios/total_ratio

        R = np.array(batch_datas["rewards"])
        ratios = np.reshape(ratios, [ratios.shape[0], 1])
        ratios = np.tile(ratios, [1, R.shape[1]])
        R = ratios * R

        estimated_rewards = estimated_rewards + np.sum(R, axis=0)

    print("Time elapsed:", time.time()-start_time)
    print("estimated_rewards:", estimated_rewards)

    return estimated_rewards

def offline_session(sess, agent, dataset, args, epoch=0):
    print("="*50, "offline_session testing", "="*50)
    '''
    用于Session结构的测试评测,采用数据结构2的类型,即拿去用户的所有访问的page信息,直接输出重排序列
    '''
    probs = []
    labels = []
    first = False
    k_scope = [3,5,10,12]
    auc_k = []
    init_auc_k = []
    ndcg_k = []
    map_k = []
    init_ndcg_k = []
    init_map_k = []
    ini_scores = []
    
    
    # if args.is_sequence or args.full_sequence:
    for _, eval_datas in AvitoSessionKTimeActorDataInput(dataset, args.batch_size):
        batch_datas = get_aggregated_batch4avito_session_k_seq(eval_datas)
        policy_action = agent.select_action(sess, batch_datas)
        probs.extend(policy_action)
        ini_scores.extend(batch_datas["actions"]["single_score_label"])
        # labels.extend(batch_datas["actions"]["ctr_label"])
        labels.extend(batch_datas["actions"]["single_ctr_label"])
        if first:
            first = False
            print("prob:", probs)
            print("label:", labels)

        print('EVAL AC:')
        batch_auc = calculate_bacth_prob_metrix(policy_action, batch_datas["actions"]["single_ctr_label"])
        batch_init_auc = calculate_bacth_prob_metrix(batch_datas["actions"]["single_score_label"],
                                                     batch_datas["actions"]["single_ctr_label"])

        auc_k.append(batch_auc)
        init_auc_k.append(batch_init_auc)
    # else:
    #     for user_pvid, eval_datas in CBUSessionActorDataInput(dataset, args.batch_size):
    #         batch_datas = get_aggregated_batch4cbu_session(eval_datas)
    #         policy_action = agent.select_action(sess, batch_datas)
    #         probs.extend(policy_action)
    #         labels.extend(batch_datas["actions"]["ctr_label"])
    #         if first:
    #             first = False
    #             print("prob:", probs)
    #             print("label:", labels)
    #
    #         if args.show_item_index:
    #             item_spare = batch_datas["states"]["item_spare_feature"]
    #             for i in range(len(item_spare)):
    #                 print("user/pvid:", user_pvid[i])
    #                 origin_list = item_spare[i, :, 0]
    #                 print("Origin item list:", origin_list)
    #                 final_list = sorted(range(12), key=lambda k: policy_action[i][k])
    #                 print("Final item list:", np.array(origin_list)[final_list].tolist())
    #                 print("Origin click:", batch_datas["actions"]["ctr_label"][i])
    #
    #         batch_auc = calculate_bacth_prob_metrix(policy_action, batch_datas["actions"]["ctr_label"])
    #         batch_init_auc = calculate_bacth_prob_metrix(batch_datas["actions"]["score_label"], batch_datas["actions"]["ctr_label"])
    #         mean_ndcg, mean_map = evaluate_matrix(batch_preds=policy_action, batch_labels=batch_datas["actions"]["ctr_label"])
    #         init_ndcg, init_map = evaluate_matrix(batch_preds=batch_datas["actions"]["score_label"], batch_labels=batch_datas["actions"]["ctr_label"])
    #
    #         auc_k.append(batch_auc)
    #         init_auc_k.append(batch_init_auc)
    #
    #         ndcg_k.append(mean_ndcg)
    #         map_k.append(mean_map)
    #         init_ndcg_k.append(init_ndcg)
    #         init_map_k.append(init_map)

    
    # k_scope = [1,3,5,10,12]
    # aucs, init_aucs = calculate_bacth_prob_metrix_topk(probs, labels, k_scope)
    mean_ndcg, mean_map = evaluate_matrix_multi(batch_preds=probs, batch_labels=labels)
    init_ndcg, init_map = evaluate_matrix_multi(batch_preds=ini_scores, batch_labels=labels)
    for index, _k in enumerate(k_scope):
        to_show = "|EPOCH: {}| AUC:  {}| ORI_AUC: {} |Top{}:| NDCG: {}| MAP: {}| ORI_NDCG: {}| ORI_MAP: {}|".format(
            epoch, np.mean(auc_k), np.mean(init_auc_k),
            _k, mean_ndcg[index], mean_map[index],
            init_ndcg[index], init_map[index])
        # print("-" * len(to_show))
        print(to_show)
        print("-" * len(to_show))
    return np.mean(auc_k)
    # for index, _k in enumerate(k_scope):
    #     print("@{}:  AUC {}  ORI_AUC {}".format(_k, np.mean(auc_k[_k]), np.mean(init_auc_k[_k])))

def get_batch(data, batch_size, batch_no):
    return data[batch_size * batch_no: batch_size * (batch_no + 1)]

def run(train_dateset, test_dateset, ac_train_dataset, ac_eval_dataset, args):
    k_scope = [3, 5, 10, 12]
    model_name = '{}'.format(args.algo)

    total_model_path = os.path.join(buckets, 'total')
    sl_model_path = os.path.join(buckets, 'sl')
    ac_model_path = os.path.join(buckets, 'ac')

    if not os.path.exists(sl_model_path):
        os.makedirs(sl_model_path)
    if not os.path.exists(ac_model_path):
        os.makedirs(ac_model_path)
    if not os.path.exists(total_model_path):
        os.makedirs(total_model_path)
    save_name = 'save_ac_model_{}_{}_ckpt'.format(model_name, "session")
    save_sl_name = 'save_sl_model_{}_{}_ckpt'.format(model_name, "session")
    save_total_name = 'save_total_model_{}_{}_ckpt'.format(model_name, "session")
    save_path = os.path.join(ac_model_path, save_name)
    save_sl_path = os.path.join(sl_model_path, save_sl_name)
    save_total_path = os.path.join(total_model_path, save_total_name)
    load_sl_dir = sl_model_path
    load_ac_dir = ac_model_path
    load_total_dir = total_model_path
    print("save model path", save_path)
    print("save sl model path", save_sl_path)
    print("save total model path", save_total_path)

    tf.reset_default_graph()
    agent = create_policy(args)

    # gpu settings
    gpu_options = tf.GPUOptions(allow_growth=True)

    training_monitor = {
        'actor_loss': [],
        'critic_loss': [],
        'max_rewards': []
    }
    
    # training process
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        start_time = time.time()

        if args.add_train:
            agent.sl_ac_load(sess, load_sl_dir, load_ac_dir)

        if args.train_sl:
            print("="*50,"Train SL model","="*50)
            data_size = len(train_dateset[0])
            batch_num = data_size // args.batch_size
            eval_iter_num = (data_size // 5) // args.batch_size
            print('train SL', data_size, batch_num)
            for e in range(int(args.bc_steps)):
                train_losses = []
                for _, train_datas in AvitoSessionSLDataInput(train_dateset, args.batch_size):
                    batch_datas = get_aggregated_batch4avito_session_sl(train_datas)
                    train_loss = agent.sl_train(sess, batch_datas)
                    train_losses.append(train_loss)

                if e % 2 == 0:
                    losses = []
                    probs = []
                    labels = []
                    auc_12 = []
                    init_auc_12 = []
                    print('EVAL SL:')
                    for _, eval_datas in AvitoSessionSLDataInput(test_dateset, args.batch_size):
                        eval_batch_datas = get_aggregated_batch4avito_session_sl(eval_datas)
                        loss, prob, label = agent.sl_eval(sess, eval_batch_datas)
                        losses.append(loss)
                        probs.extend(prob)
                        labels.extend(label)
                        # ori_probs = eval_batch_datas["score_label"]
                        
                        _auc_12 = calculate_bacth_prob_metrix(prob, label)

                        _init_auc_12 = 0.5 #calculate_bacth_prob_metrix(ori_probs, label)
                        auc_12.append(_auc_12)
                        init_auc_12.append(_init_auc_12)
                    print('labels:', np.array(labels).shape)
                    mean_ndcg, mean_map = evaluate_matrix_multi(batch_preds=probs, batch_labels=labels)
                    for index, _k in enumerate(k_scope):
                        to_show = "EPOCH: ", e, "; TRAIN LOSS:", np.mean(train_losses), "; EVAL LOSS:", np.mean(
                            losses), "; AUC:", np.mean(auc_12), "; ORI_AUC:", np.mean(
                            init_auc_12), "; TOP:", _k, "; NDCG:", mean_ndcg[index], "; MAP:", mean_map[index]
                        print("-" * len(to_show))
                        print(to_show)
                        print("-" * len(to_show))
                    


                    if args.save_sl:
                        agent.sl_save(sess, save_sl_path)

            print("Finish the train SL model... Stopping Process... Begin Train AC Model...")

        max_auc_ac = 0.5
        for e in range(int(args.max_steps)):
            total_actor_loss = []
            total_critic_loss = []
            # if args.is_sequence or args.full_sequence:
            for _, train_datas in AvitoSessionKTimeActorDataInput(ac_train_dataset, args.batch_size):
                batch_datas = get_aggregated_batch4avito_session_k_seq(train_datas)

                actor_loss, critic_loss = agent.train(sess, batch_datas)
                agent.soft_update(sess)
                training_monitor["actor_loss"].append(actor_loss)
                training_monitor["critic_loss"].append(critic_loss)

                total_actor_loss.append(actor_loss)
                total_critic_loss.append(critic_loss)
            # else:
            #     for _, train_datas in CBUSessionActorDataInput(ac_train_dataset, args.batch_size):
            #         batch_datas = get_aggregated_batch4cbu_session(train_datas)
            #
            #         actor_loss, critic_loss = agent.train(sess, batch_datas)
            #         agent.soft_update(sess)
            #         training_monitor["actor_loss"].append(actor_loss)
            #         training_monitor["critic_loss"].append(critic_loss)
            #
            #         total_actor_loss.append(actor_loss)
            #         total_critic_loss.append(critic_loss)

            if e % args.save_every == 0:
                print('At ieration', e, 'time elapsed', time.time() - start_time,'mean actor loss:', np.mean(actor_loss), 'mean critic loss :', np.mean(critic_loss))
                # estimated_rewards = offline_ab(sess, agent, test_dateset, args)
                res_auc = offline_session(sess, agent, ac_eval_dataset, args, e)
                print("MAX AUC: {}, CURRENT AUC: {}".format(max_auc_ac, res_auc))
                if args.save_model:
                    if res_auc > max_auc_ac:
                        max_auc_ac = res_auc
                        print("Get the best AUC save the model...")
                        agent.one_stage_save(sess, save_path)
                        # agent.total_save(sess, save_total_name)
                    elif(e % (args.save_every * 5) == 0):
                        print("Every epoch {} save the model...".format(args.save_every * 5))
                        agent.one_stage_save(sess, save_path)
                        # agent.total_save(sess, save_total_name)
                

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default="test", type=str, help='exp name')
    parser.add_argument('--setting_path', type=str, default='./configs/multi_critic_avito_session.json', help='setting dir')
    parser.add_argument('--add_train', type=bool, default=False, help='add train info')
    FLAGS, _ = parser.parse_known_args()
    return FLAGS

def load_file(save_file):
    with open(save_file, 'r') as f:
        data = f.readlines()
    records = []
    for line in data:
        records.append([eval(v) for v in line.split('\t')])
    return records

if __name__ == '__main__':
    # parameters
    random.seed(1234)
    args = parse_args()
    if args.setting_path:
        args = load_parse_from_json(args, args.setting_path)
    processed_dir = "./avito"
    tf.app.flags.DEFINE_string("tables", "", "tables info")
    tf.app.flags.DEFINE_string("buckets", "", "buckets info")
    # tf.app.flags.DEFINE_bool("add_train", False, "add train info")
    FLAGS = tf.app.flags.FLAGS
    print("tables:" + FLAGS.tables)
    # 0 train 1 small train 2 test 
    tables = FLAGS.tables
    tables = FLAGS.tables.split(",")
    print("split tables", tables)

    # args.data_set_name = tables[0].split("/")[-2]
    args.add_train = False

    # FLAGS = tf.app.flags.FLAGS
    buckets = FLAGS.buckets
    print("buckets:", buckets)

    print("="*50, "load train dataset", "="*50)

    m = 0.1

    # construct training files
    train_dir = os.path.join(processed_dir, 'data.train')

    train_lists = None
    if os.path.isfile(train_dir):
        # train_lists = pkl.load(open(train_dir, 'rb'))
        train_lists = load_file(train_dir)

    # construct test files
    test_lists = None
    test_dir = os.path.join(processed_dir, 'data.test')
    if os.path.isfile(test_dir):
        # test_lists = pkl.load(open(test_dir, 'rb'))
        test_lists = load_file(test_dir)
    # ===== training =====

    run(train_lists, test_lists, train_lists, test_lists, args)