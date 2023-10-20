#coding=utf-8
import numpy as np
import time
import datetime
import tensorflow as tf
import time

def get_aggregated_batch4cbu_session_k_seq(batch_data):
    buffers = batch_data
    # print("batch data:", batch_data)
    batch_size = len(buffers)
    buffers_used_keys = {
        "states": ["user_spare_feature", "user_dense_feature", "item_spare_feature", "item_dense_feature", "hist_spare_feature"],
        "actions": ["ctr_label", "conver_label", "score_label", "single_ctr_label", "single_score_label"],
        "rewards": ["ctr_rewards", "conver_rewards", "score_rewards"],
        "next_states": ["user_spare_feature", "user_dense_feature", "item_spare_feature", "item_dense_feature", "hist_spare_feature"],
        "dones": ["done"],
        "time": ["time"]
    }
    buffers_used_keys_size = {
        "states": [8, 14, 37, 29, 7],
        "actions": [12, 12, 12],
        "rewards": [1, 1, 1],
        "next_states": [8, 14, 37, 29, 7],
        "dones": [1]
    }
    buffers_used_keys_total_size = {
        "user_spare_feature": 8, 
        "user_dense_feature": 14, 
        "item_spare_feature": 37, 
        "item_dense_feature": 29, 
        "hist_spare_feature": 7,
        "ctr_label": 12,
        "conver_label": 12,
        "score_label": 12,
        "single_ctr_label": 12,
        "single_score_label": 12,
        "done": 1,
        "time": 1
    }
    # ["user_spare_feature", "user_dense_feature", "item_spare_feature", "item_dense_feature", "hist_spare_feature", "cur_mask", "pre_item_click_spare_feature", "done"]
    # ["ctr_label", "conver_label", "final_score"]
    result = {}

    for index, item in enumerate(buffers_used_keys.keys()):
        sub_item_info = []
        if("state" in item):
            res_item = {}
            for index, sub_item in enumerate(buffers_used_keys[item]):
                if "item" in sub_item:
                    sub_item_info = np.array([buffers[i][sub_item] for i in range(len(buffers))]).reshape(batch_size, 3, 12, buffers_used_keys_total_size[sub_item])
                    if "next" in item:
                        next_sub_item_info = sub_item_info[1:]
                        add_item_info = sub_item_info[-1].reshape(1, 3, 12, buffers_used_keys_total_size[sub_item])
                        next_sub_item_info = np.concatenate([next_sub_item_info, add_item_info], axis=0)
                        for index in range(1, len(buffers)):
                            if buffers[index]["done"]:
                                next_sub_item_info[index] = next_sub_item_info[index-1]
                        sub_item_info = next_sub_item_info
                if "hist" in sub_item:
                    sub_item_info = np.array([buffers[i][sub_item] for i in range(len(buffers))]).reshape(batch_size, 50, buffers_used_keys_total_size[sub_item])
                    add_info = np.zeros([batch_size, 12, buffers_used_keys_total_size[sub_item]])
                    sub_item_info = np.concatenate([sub_item_info, add_info], axis=1)
                    if "next" in item:
                        next_sub_item_info = sub_item_info[1:]
                        add_item_info = sub_item_info[-1].reshape(1, 62, buffers_used_keys_total_size[sub_item])
                        next_sub_item_info = np.concatenate([next_sub_item_info, add_item_info], axis=0)
                        for index in range(1, len(buffers)):
                            if buffers[index]["done"]:
                                next_sub_item_info[index] = next_sub_item_info[index-1]
                        sub_item_info = next_sub_item_info
                if "user" in sub_item:
                    sub_item_info = np.array([buffers[i][sub_item] for i in range(len(buffers))]).reshape(batch_size, buffers_used_keys_total_size[sub_item])
                res_item[sub_item] = sub_item_info
        
        if("actions" in item):
            res_item = {}
            for index, sub_item in enumerate(buffers_used_keys[item]):
                if "single" in sub_item:
                    sub_item_info = np.array([buffers[i][sub_item] for i in range(len(buffers))]).reshape(batch_size, buffers_used_keys_total_size[sub_item])
                    res_item[sub_item] = sub_item_info
                else:
                    sub_item_info = np.array([buffers[i][sub_item] for i in range(len(buffers))]).reshape(batch_size, 3, buffers_used_keys_total_size[sub_item])
                    res_item[sub_item] = sub_item_info
        
        if("rewards" in item):
            ctr_sub_item_info = np.array([buffers[i]["ctr_label"] for i in range(len(buffers))]).reshape(batch_size, 3, buffers_used_keys_total_size["ctr_label"])
            ctr_sub_item_info = np.sum(ctr_sub_item_info, axis=-1)
            conver_sub_item_info = np.array([buffers[i]["conver_label"] for i in range(len(buffers))]).reshape(batch_size, 3, buffers_used_keys_total_size["conver_label"])
            conver_sub_item_info = np.sum(conver_sub_item_info, axis=-1)
            score_sub_item_info = np.array([buffers[i]["score_label"] for i in range(len(buffers))]).reshape(batch_size, 3, buffers_used_keys_total_size["score_label"])
            score_sub_item_info = np.sum(score_sub_item_info, axis=-1)
            res_item = {
                "ctr_rewards": ctr_sub_item_info,
                "conver_rewards": conver_sub_item_info,
                "score_rewards": score_sub_item_info
            }
        
        if("done" in item):
            res_item = {}
            for index, sub_item in enumerate(buffers_used_keys[item]):
                sub_item_info = np.array([buffers[i][sub_item] for i in range(len(buffers))]).reshape(batch_size, buffers_used_keys_total_size[sub_item])
                res_item[sub_item] = 1 - sub_item_info
        
        if("time" in item):
            res_item = {}
            for index, sub_item in enumerate(buffers_used_keys[item]):
                sub_item_info = np.array([buffers[i][sub_item] for i in range(len(buffers))]).reshape(batch_size, 3, buffers_used_keys_total_size[sub_item])
                res_item[sub_item] = sub_item_info

        result[item] = res_item

        # print("result:", result)

    return result


def get_aggregated_batch4avito_session_k_seq(batch_data):
    buffers = batch_data
    # print("batch data:", batch_data)
    batch_size = len(buffers)
    buffers_used_keys = {
        "states": ["user_spare_feature", "user_dense_feature", "item_spare_feature", "item_dense_feature",
                   "hist_spare_feature"],
        "actions": ["ctr_label", "conver_label", "score_label", "single_ctr_label", "single_score_label"],
        "rewards": ["ctr_rewards", "conver_rewards", "score_rewards"],
        "next_states": ["user_spare_feature", "user_dense_feature", "item_spare_feature", "item_dense_feature",
                        "hist_spare_feature"],
        "dones": ["done"],
        "time": ["time"]
    }
    buffers_used_keys_size = {
        "states": [8, 14, 37, 29, 7],
        "actions": [12, 12, 12],
        "rewards": [1, 1, 1],
        "next_states": [8, 14, 37, 29, 7],
        "dones": [1]
    }
    buffers_used_keys_total_size = {
        "user_spare_feature": 4,
        "user_dense_feature": 4,
        "item_spare_feature": 2,
        "item_dense_feature": 2,
        "hist_spare_feature": 2,
        "ctr_label": 12,
        "conver_label": 12,
        "score_label": 12,
        "single_ctr_label": 12,
        "single_score_label": 12,
        "done": 1,
        "time": 1
    }
    # ["user_spare_feature", "user_dense_feature", "item_spare_feature", "item_dense_feature", "hist_spare_feature", "cur_mask", "pre_item_click_spare_feature", "done"]
    # ["ctr_label", "conver_label", "final_score"]
    result = {}

    for index, item in enumerate(buffers_used_keys.keys()):
        sub_item_info = []
        if ("state" in item):
            res_item = {}
            for index, sub_item in enumerate(buffers_used_keys[item]):
                if "item" in sub_item:
                    sub_item_info = np.array([buffers[i][sub_item] for i in range(len(buffers))]).reshape(batch_size, 3,
                                                                                                          12,
                                                                                                          buffers_used_keys_total_size[
                                                                                                              sub_item])
                    if "next" in item:
                        next_sub_item_info = sub_item_info[1:]
                        add_item_info = sub_item_info[-1].reshape(1, 3, 12, buffers_used_keys_total_size[sub_item])
                        next_sub_item_info = np.concatenate([next_sub_item_info, add_item_info], axis=0)
                        for index in range(1, len(buffers)):
                            if buffers[index]["done"]:
                                next_sub_item_info[index] = next_sub_item_info[index - 1]
                        sub_item_info = next_sub_item_info
                if "hist" in sub_item:
                    sub_item_info = np.array([buffers[i][sub_item] for i in range(len(buffers))]).reshape(batch_size,
                                                                                                          12,
                                                                                                          buffers_used_keys_total_size[
                                                                                                              sub_item])
                    add_info = np.zeros([batch_size, 12, buffers_used_keys_total_size[sub_item]])
                    sub_item_info = np.concatenate([sub_item_info, add_info], axis=1)
                    if "next" in item:
                        next_sub_item_info = sub_item_info[1:]
                        add_item_info = sub_item_info[-1].reshape(1, 12+12, buffers_used_keys_total_size[sub_item])
                        next_sub_item_info = np.concatenate([next_sub_item_info, add_item_info], axis=0)
                        for index in range(1, len(buffers)):
                            if buffers[index]["done"]:
                                next_sub_item_info[index] = next_sub_item_info[index - 1]
                        sub_item_info = next_sub_item_info
                if "user" in sub_item:
                    sub_item_info = np.array([buffers[i][sub_item] for i in range(len(buffers))]).reshape(batch_size,
                                                                                                          buffers_used_keys_total_size[
                                                                                                              sub_item])
                res_item[sub_item] = sub_item_info

        if ("actions" in item):
            res_item = {}
            for index, sub_item in enumerate(buffers_used_keys[item]):
                if "single" in sub_item:
                    sub_item_info = np.array([buffers[i][sub_item] for i in range(len(buffers))]).reshape(batch_size,
                                                                                                          buffers_used_keys_total_size[sub_item])
                    res_item[sub_item] = sub_item_info
                else:
                    sub_item_info = np.array([buffers[i][sub_item] for i in range(len(buffers))]).reshape(batch_size, 3,
                                                                                                          buffers_used_keys_total_size[sub_item])
                    res_item[sub_item] = sub_item_info

        if ("rewards" in item):
            ctr_sub_item_info = np.array([buffers[i]["ctr_label"] for i in range(len(buffers))]).reshape(batch_size, 3,
                                                                                                         buffers_used_keys_total_size[
                                                                                                             "ctr_label"])
            ctr_sub_item_info = np.sum(ctr_sub_item_info, axis=-1)
            conver_sub_item_info = np.array([buffers[i]["conver_label"] for i in range(len(buffers))]).reshape(
                batch_size, 3, buffers_used_keys_total_size["conver_label"])
            conver_sub_item_info = np.sum(conver_sub_item_info, axis=-1)
            score_sub_item_info = np.array([buffers[i]["score_label"] for i in range(len(buffers))]).reshape(batch_size,
                                                                                                             3,
                                                                                                             buffers_used_keys_total_size[
                                                                                                                 "score_label"])
            score_sub_item_info = np.sum(score_sub_item_info, axis=-1)
            res_item = {
                "ctr_rewards": ctr_sub_item_info,
                "conver_rewards": conver_sub_item_info,
                "score_rewards": score_sub_item_info
            }

        if ("done" in item):
            res_item = {}
            for index, sub_item in enumerate(buffers_used_keys[item]):
                sub_item_info = np.array([buffers[i][sub_item] for i in range(len(buffers))]).reshape(batch_size,
                                                                                                      buffers_used_keys_total_size[
                                                                                                          sub_item])
                res_item[sub_item] = 1 - sub_item_info

        if ("time" in item):
            res_item = {}
            for index, sub_item in enumerate(buffers_used_keys[item]):
                sub_item_info = np.array([buffers[i][sub_item] for i in range(len(buffers))]).reshape(batch_size, 3,
                                                                                                      buffers_used_keys_total_size[
                                                                                                          sub_item])
                res_item[sub_item] = sub_item_info

        result[item] = res_item

        # print("result:", result)

    return result

def get_aggregated_batch4cbu_session(batch_data):
    buffers = batch_data
    batch_size = len(buffers)
    buffers_used_keys = {
        "states": ["user_spare_feature", "user_dense_feature", "item_spare_feature", "item_dense_feature", "hist_spare_feature"],
        "actions": ["ctr_label", "conver_label", "score_label"],
        "rewards": ["ctr_rewards", "conver_rewards", "score_rewards"],
        "next_states": ["user_spare_feature", "user_dense_feature", "item_spare_feature", "item_dense_feature", "hist_spare_feature"],
        "dones": ["done"]
    }
    buffers_used_keys_size = {
        "states": [8, 14, 37, 29, 7],
        "actions": [12, 12, 12],
        "rewards": [1, 1, 1],
        "next_states": [8, 14, 37, 29, 7],
        "dones": [1]
    }
    buffers_used_keys_total_size = {
        "user_spare_feature": 8, 
        "user_dense_feature": 14, 
        "item_spare_feature": 37, 
        "item_dense_feature": 29, 
        "hist_spare_feature": 7,
        "ctr_label": 12,
        "conver_label": 12,
        "score_label": 12,
        "done": 1
    }
    # ["user_spare_feature", "user_dense_feature", "item_spare_feature", "item_dense_feature", "hist_spare_feature", "cur_mask", "pre_item_click_spare_feature", "done"]
    # ["ctr_label", "conver_label", "final_score"]
    result = {}

    for index, item in enumerate(buffers_used_keys.keys()):
        sub_item_info = []
        if("state" in item):
            res_item = {}
            for index, sub_item in enumerate(buffers_used_keys[item]):
                if "item" in sub_item:
                    sub_item_info = np.array([buffers[i][sub_item] for i in range(len(buffers))]).reshape(batch_size, 12, buffers_used_keys_total_size[sub_item])
                if "hist" in sub_item:
                    sub_item_info = np.array([buffers[i][sub_item] for i in range(len(buffers))]).reshape(batch_size, 62, buffers_used_keys_total_size[sub_item])
                if "user" in sub_item:
                    sub_item_info = np.array([buffers[i][sub_item] for i in range(len(buffers))]).reshape(batch_size, buffers_used_keys_total_size[sub_item])
                res_item[sub_item] = sub_item_info
        
        if("actions" in item):
            res_item = {}
            for index, sub_item in enumerate(buffers_used_keys[item]):
                sub_item_info = np.array([buffers[i][sub_item] for i in range(len(buffers))]).reshape(batch_size, buffers_used_keys_total_size[sub_item])
                res_item[sub_item] = sub_item_info
        
        if("rewards" in item):
            ctr_sub_item_info = np.array([buffers[i]["ctr_label"] for i in range(len(buffers))]).reshape(batch_size, buffers_used_keys_total_size["ctr_label"])
            ctr_sub_item_info = np.sum(ctr_sub_item_info, axis=-1)
            conver_sub_item_info = np.array([buffers[i]["conver_label"] for i in range(len(buffers))]).reshape(batch_size, buffers_used_keys_total_size["conver_label"])
            conver_sub_item_info = np.sum(conver_sub_item_info, axis=-1)
            score_sub_item_info = np.array([buffers[i]["score_label"] for i in range(len(buffers))]).reshape(batch_size, buffers_used_keys_total_size["score_label"])
            score_sub_item_info = np.sum(score_sub_item_info, axis=-1)
            res_item = {
                "ctr_rewards": ctr_sub_item_info,
                "conver_rewards": conver_sub_item_info,
                "score_rewards": score_sub_item_info
            }
        
        if("done" in item):
            res_item = {}
            for index, sub_item in enumerate(buffers_used_keys[item]):
                sub_item_info = np.array([buffers[i][sub_item] for i in range(len(buffers))]).reshape(batch_size, buffers_used_keys_total_size[sub_item])
                res_item[sub_item] = 1 - sub_item_info

        result[item] = res_item

        # print("result:", result)

    return result

def get_aggregated_batch4cbu_session_sl(batch_data):
    buffers = batch_data
    batch_size = len(buffers)

    buffers_used_keys = ["user_spare_feature", "user_dense_feature", "item_spare_feature", "item_dense_feature", "hist_spare_feature", "cur_mask", "ctr_label", "score_label"]
    buffers_used_keys_total_size = {
        "user_spare_feature": 8,
        "user_dense_feature": 14,
        "item_spare_feature": 12*37,
        "item_dense_feature": 12*29,
        "hist_spare_feature": 50*7,
        "ctr_label": 12,
        "cur_mask": 12,
        "score_label": 12
    }

    result = {}
    for index, item in enumerate(buffers_used_keys):
        item_info = np.array([buffers[i][item] for i in range(len(buffers))]).reshape(batch_size, buffers_used_keys_total_size[item])
        res_item = item_info

        if item == "item_spare_feature":
            res_item = item_info.reshape(batch_size, 12, buffers_used_keys_total_size[item]/12)
        if item == "item_dense_feature":
            res_item = item_info.reshape(batch_size, 12, buffers_used_keys_total_size[item]/12)

        if item == "hist_spare_feature":
            final_item = np.zeros((batch_size, 62, 7))
            res_item = item_info.reshape(batch_size, 50, buffers_used_keys_total_size[item]/50)
            final_item[:, :50, :] = res_item
            res_item = final_item

        result[item] = res_item

    return result

def get_aggregated_batch4avito_session_sl(batch_data):
    buffers = batch_data
    batch_size = len(buffers)

    buffers_used_keys = ["user_spare_feature", "user_dense_feature", "item_spare_feature", "item_dense_feature", "hist_spare_feature", "ctr_label"]
    buffers_used_keys_total_size = {
        "user_spare_feature": 4,
        "user_dense_feature": 4,
        "item_spare_feature": 12*2,
        "item_dense_feature": 12*2,
        "hist_spare_feature": 12*2,
        "ctr_label": 12
    }

    result = {}
    for index, item in enumerate(buffers_used_keys):
        item_info = np.array([buffers[i][item] for i in range(len(buffers))]).reshape(batch_size, buffers_used_keys_total_size[item])
        res_item = item_info

        if item == "item_spare_feature":
            res_item = item_info.reshape(batch_size, 12, buffers_used_keys_total_size[item]/12)
        if item == "item_dense_feature":
            res_item = item_info.reshape(batch_size, 12, buffers_used_keys_total_size[item]/12)

        if item == "hist_spare_feature":
            final_item = np.zeros((batch_size, 24, 2))
            res_item = item_info.reshape(batch_size, 12, buffers_used_keys_total_size[item]/12)
            final_item[:, :12, :] = res_item
            res_item = final_item

        result[item] = res_item

    return result

def get_aggregated_batch4cbu_offline(batch_data):
    buffers = batch_data
    batch_size = len(buffers)

    buffers_used_keys = ["user_spare", "user_dense", "item_spare", "item_dense", "hist_spare", "ctr_label", "conver_label"]
    buffers_used_keys_total_size = {
        "user_spare": 8, 
        "user_dense": 14, 
        "item_spare": 12*37, 
        "item_dense": 12*29, 
        "hist_spare": 50*7, 
        "ctr_label": 12, 
        "conver_label": 12
    }

    result = {}
    for index, item in enumerate(buffers_used_keys):
        item_info = np.array([buffers[i][item] for i in range(len(buffers))]).reshape(batch_size, buffers_used_keys_total_size[item])
        res_item = item_info

        if item == "item_spare":
            res_item = item_info.reshape(batch_size, 12, buffers_used_keys_total_size[item]/12)
        if item == "item_dense":
            res_item = item_info.reshape(batch_size, 12, buffers_used_keys_total_size[item]/12)
        if item == "hist_spare":
            res_item = item_info.reshape(batch_size, 50, buffers_used_keys_total_size[item]/50)


        result[item] = res_item
    return result

def get_aggregated_batch4cbu_split(batch_data):
    buffers = batch_data
    batch_size = len(buffers)
    
    buffers_used_keys = {
        "states": ["user_spare_feature", "user_dense_feature", "item_spare_feature", "item_dense_feature", "dynamic_item_spare_feature", "dynamic_item_dense_feature", "hist_spare_feature"],
        "actions": ["actions"],
        "rewards": ["rewards"],
        "next_states": ["user_spare_feature", "user_dense_feature", "item_spare_feature", "item_dense_feature", "dynamic_item_spare_feature", "dynamic_item_dense_feature", "hist_spare_feature"],
        "dones": ["dones"]
    }
    buffers_used_keys_size = {
        "states": [8, 14, 37, 29, 7],
        "actions": [1],
        "rewards": [2],
        "next_states": [8, 14, 37, 29, 7],
        "dones": [1]
    }
    user_spare_start, user_spare_end = 0, 8
    user_dense_start, user_dense_end = 8, 8+14
    item_feature_start, item_feature_end = 8+14, 8+14+12*66
    dynamic_item_feature_start, dynamic_item_feature_end = 8+14+12*66, 8+14+12*66+12*66
    hist_feature_start, hist_feature_end = 8+14+12*66+12*66, 8+14+12*66+12*66+50*7
    item_spare_feature_index = list(range(23)) + [27] + list(range(30,39)) + list(range(59,63))
    dense_feature_index = [i for i in range(66) if i not in item_spare_feature_index]
    
    # buffers_used_keys_total_size = [22 + 24 * 66 + 50 * 7, 1, 2, 22 + 24 * 66 + 50 * 7, 1]
    buffers_used_keys_total_size = {
        "states": 22 + 24 * 66 + 50 * 7,
        "actions": 1,
        "rewards": 2,
        "next_states": 22 + 24 * 66 + 50 * 7,
        "dones": 1
    }
    result = {}
    for index, item in enumerate(buffers_used_keys.keys()):
        if not isinstance(buffers[0][item], int):
            assert len(buffers[0][item]) == buffers_used_keys_total_size[item], "Index {} Item: {} Length is {} Have Error Length {}".format(index, item, len(buffers[0][item]), buffers_used_keys_total_size[item])
        item_info = np.array([buffers[i][item] for i in range(len(buffers))]).reshape(batch_size, buffers_used_keys_total_size[item])
        res_item = item_info
        if("state" in item):
            res_item = {}
            for index, sub_item in enumerate(buffers_used_keys[item]):
                if sub_item == "user_spare_feature":
                    res_item[sub_item] = item_info[:, user_spare_start:user_spare_end].astype(float).astype(int)  # batch * 8
                if sub_item == "user_dense_feature":
                    res_item[sub_item] = item_info[:, user_dense_start:user_dense_end].astype(float) # batch * 14
                if sub_item == "item_spare_feature":
                    tmp = item_info[:, item_feature_start: item_feature_end].reshape(batch_size, 12, 66).astype(float).astype(int)
                    res_item[sub_item] = tmp[:, :, item_spare_feature_index] # batch * 12 * 37
                if sub_item == "item_dense_feature":
                    tmp = item_info[:, item_feature_start: item_feature_end].reshape(batch_size, 12, 66).astype(float)
                    res_item[sub_item] = tmp[:, :, dense_feature_index] # batch * 12 * 29
                if sub_item == "dynamic_item_spare_feature":
                    tmp = item_info[:, dynamic_item_feature_start: dynamic_item_feature_end].reshape(batch_size, 12, 66).astype(float).astype(int)
                    res_item[sub_item] = tmp[:, :, item_spare_feature_index] # batch * 12 * 37
                if sub_item == "dynamic_item_dense_feature":
                    tmp = item_info[:, dynamic_item_feature_start: dynamic_item_feature_end].reshape(batch_size, 12, 66).astype(float)
                    res_item[sub_item] = tmp[:, :, dense_feature_index] # batch * 12 * 29
                if sub_item == "hist_spare_feature":
                    tmp = item_info[:, hist_feature_start: hist_feature_end].reshape(batch_size, 50, 7).astype(float).astype(int)
                    res_item[sub_item] = tmp # batch * 50 * 7
            dynamic_spare = res_item["dynamic_item_spare_feature"][:, :, 0:7]
            dynamic_spare[:, :, 5:7] = 0
            res_item["hist_spare"] = np.concatenate([res_item["hist_spare_feature"], dynamic_spare], axis=1)
            # item id 取余部分
            res_item["user_spare_feature"][:, 0] = res_item["user_spare_feature"][:, 0] % 8000000
            res_item["item_spare_feature"][:, :, 0] = res_item["item_spare_feature"][:, :, 0] % 6000000
            res_item["item_spare_feature"][:, :, 3] = res_item["item_spare_feature"][:, :, 3] % 600000
            res_item["hist_spare"][:, :, 0] = res_item["hist_spare"][:, :, 0] % 6000000
            res_item["hist_spare"][:, :, 3] = res_item["hist_spare"][:, :, 3] % 600000
            # res_item["dynamic_item_spare_feature"] = []
            # res_item["dynamic_item_dense_feature"] = []
            # res_item["hist_spare_feature"] = []

        if item == "dones":
            res_item = 1 - item_info

        result[item] = res_item
    return result

def get_aggregated_batch4cbu(batch_data):
    '''
    batch
    '''
    buffers = batch_data

    buffers_used_keys = ["states", "actions", "rewards", "next_states", "dones"]
    buffers_used_keys_size = [1582, 1, 2, 1582, 1]

    batch_size = len(buffers)

    result = {}

    # print("buffer:", buffers)
    # for item in buffers_used_keys:
    #     print("Item ", item, " 's shape is:", len(buffers[0][item]))

    for index, item in enumerate(buffers_used_keys):
        res_item = np.array([buffers[i][item] for i in range(len(buffers))]).reshape(batch_size, buffers_used_keys_size[index])
        # print(item, " shape is:", res_item.shape)
        if item == "dones":
            res_item = 1 - res_item
        result[item] = res_item

    return result


class DataInput:
    def __init__(self, data, batch_size):

        self.batch_size = batch_size
        self.state_dim = 174  # user feature: 24  cancadite videos: 150 * 6(feature length)  history video 20 * 6(feature length)
        self.reward_dim = 2
        self.data = data
        # self.sess = sess
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0
        self.test = False
        '''
        B: batch

        '''
        
        self.buffer_name = ["states", "next_states", "actions", "rewards", "dones"]
        # print("epoch_size", self.epoch_size)

    def __iter__(self):
        return self

    def next(self):

        if self.i == self.epoch_size:
            raise StopIteration

        buffers = []

        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size,
                                                     len(self.data))]
        for line in ts:
            if self.test:
                print(line)
                self.test = False
            try:
                # print("line length:", len(line))
                states = line[0].split(",")
                # print("len value:", len(states))
                # print("value:", states)
                next_states = line[3].split(",")
                actions = line[1]
                rewards = line[2].split(",")
                dones = line[4]
                buffers.append({
                    "states": np.array(states).astype(float).tolist(),
                    "next_states": np.array(next_states).astype(float).tolist(), 
                    "actions": np.array(actions).astype(int).tolist(),
                    "rewards": np.array(rewards).astype(float).tolist(), 
                    "dones": np.array(dones).astype(int).tolist()
                })
            except Exception as e:
                # print("error", line)
                print(e)
                continue

        self.i += 1

        return 0, buffers
    
class OfflineInput:
    def __init__(self, data, batch_size):

        self.batch_size = batch_size
        self.state_dim = 174  # user feature: 24  cancadite videos: 150 * 6(feature length)  history video 20 * 6(feature length)
        self.reward_dim = 2
        self.data = data
        # self.sess = sess
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0
        self.test = False
        '''
        B: batch

        '''
        
        self.buffer_name = ["user_feature", "item_feature", "hist_feature", "ctr_label", "comver_label"]
        # print("epoch_size", self.epoch_size)
        self.item_spare_feature_index = list(range(23)) + [27] + list(range(30,39)) + list(range(59,63))
        self.dense_feature_index = [i for i in range(67) if i not in self.item_spare_feature_index]

    def __iter__(self):
        return self
    
    def next(self):

        if self.i == self.epoch_size:
            raise StopIteration

        buffers = []

        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size,
                                                     len(self.data))]
        for line in ts:
            if self.test:
                print(line)
                self.test = False
            try:
                # print("line length:", len(line))
                user_feature = np.array(line[0].split(","))
                # print("len value:", len(states))
                # print("value:", states)
                item_feature = np.array(line[1].split(",")).reshape(12, 66)
                hist_feature = np.array(line[2].split(",")).astype(int).reshape(50, 7)
                ctr_label = np.array(line[3].split(",")).astype(float)
                conver_label = np.array(line[4].split(",")).astype(float)

                user_spare = user_feature[0:8].astype(int)
                user_dense = user_feature[8:].astype(float)
                item_spare = item_feature[:, self.item_spare_feature_index].astype(int)
                item_dense = item_feature[:, self.dense_feature_index].astype(float)
                buffers.append({
                    "user_spare": user_spare.tolist(),
                    "user_dense": user_dense.tolist(),
                    "item_spare": item_spare.tolist(),
                    "item_dense": item_dense.tolist(),
                    "hist_spare": hist_feature.tolist(),
                    "ctr_label": ctr_label.tolist(),
                    "conver_label": conver_label.tolist()
                })
            except Exception as e:
                # print("error", line)
                print(e)
                continue

        self.i += 1

        return 0, buffers
    
class CBUSessionSLDataInput:
    def __init__(self, data, batch_size, window_size=50, item_seq_len=12):

        self.batch_size = batch_size
        self.window_size = window_size # 历史点击序列长度
        self.item_seq_len = item_seq_len # 候选序列长度
        self.data = data
        # self.sess = sess
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0
        self.test = True
        '''
        B: batch

        SU: spare user feature
        DU: dense user feature

        T: items length
        DN: dense feature length
        SN: spare feature length

        BT: behavior length
        SBN: spare behavior feature length

        LN: ctr lable length
        '''
        # B*SU B*DU B*T*DN B*BT*SBN
        self.feature_key = ["user_spare_feature", "user_dense_feature", "item_spare_feature", "item_dense_feature", "hist_spare_feature", "cur_mask"]
        # B*LN = B*T
        self.label_key = ["ctr_label"]
        print("epoch_size", self.epoch_size)

    def __iter__(self):
        return self

    def next(self):

        if self.i == self.epoch_size:
            raise StopIteration

        buffers = []

        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size,
                                                     len(self.data))]
        feature_len = (8 + 14 + 5 + 34 + 20 + 2 + 5 + 9 + 7 * self.window_size + 36 + 3 + 4)

        spare_feature_index = list(range(23)) + [27] + list(range(30,39)) + list(range(59,63))
        dense_feature_index = [i for i in range(66) if i not in spare_feature_index]
        # print("dense len:", len(dense_feature_index)) # 29
        # print("spare len:", len(spare_feature_index)) # 36
        self.zero_count = 0
        for line in ts:
            # if self.test:
            #     print(line)
            #     self.test = False
            try:
                user_spare_profiles = []
                user_dense_profiles = []
                # cur_sapre_feature = []
                # cur_dense_feature = []
                cur_feature = []
                behavior_spare_feature = []
                cur_mask = []

                # cvr_feat_batch = []
                # score_feat_batch = []
                ctr_label = []
                score_label = []

                user_id = line[0]
                values = line[2].split(",")
                for i in range(1, self.item_seq_len + 1):
                    if not user_spare_profiles:
                        user_spare_profiles = values[(i - 1) * feature_len + 1: (i - 1) * feature_len + 1 + 8]
                    if not user_dense_profiles:
                        user_dense_profiles = values[(i - 1) * feature_len + 1 + 8: (i - 1) * feature_len + 1 + 8 + 14]
                    
                    cur_mask.append(values[i * feature_len - 3])

                    cur_feature.append(values[(i - 1) * feature_len + 1 + 8 + 14 : (i - 1) * feature_len + 1 + 8 + 14 + 1 + 65])

                    if not behavior_spare_feature:
                        behavior_spare_feature = values[(i - 1) * feature_len + 1 + 8 + 14 + 1 + 65 + 9: (i - 1) * feature_len + 1 + 8 + 14 + 1 + 65 + 9 + 7 * self.window_size]
                        final = []
                        for b_i in range(self.window_size):
                            final.append([behavior_spare_feature[b_i::50]])
                        behavior_spare_feature = final

                    ctr_label.append(values[i * feature_len - 2]) # ctr_label
                    score_label.append(values[i * feature_len - 4]) # score label
                    # print("score :", values[i * feature_len - 10: i * feature_len - 2])


                cur_feature = np.array(cur_feature).astype(float)

                buffers.append({
                    "user_spare_feature": np.array(user_spare_profiles).astype(int).tolist(),
                    "user_dense_feature": np.array(user_dense_profiles).astype(float).tolist(), 
                    "item_spare_feature": cur_feature[:, spare_feature_index].astype(int).tolist(),  # 12 * N
                    "item_dense_feature": cur_feature[:, dense_feature_index].astype(float).tolist(), # 12 * M
                    "hist_spare_feature": np.array(behavior_spare_feature).reshape(self.window_size, 7).astype(int).tolist(),  # 50 * 7
                    "cur_mask": np.array(cur_mask).astype(int).tolist(),
                    "ctr_label": np.array(ctr_label).astype(int).tolist(),
                    "score_label": np.array(score_label).astype(float).tolist()
                })
            except:
                print("error", line)
                continue

        self.i += 1
        # print("Have zeros list:", self.zero_count, " in batch size:", self.batch_size)

        return 0, buffers


class AvitoSessionSLDataInput:
    def __init__(self, data, batch_size, window_size=12, item_seq_len=12):

        self.batch_size = batch_size
        self.window_size = window_size  # 历史点击序列长度
        self.item_seq_len = item_seq_len  # 候选序列长度
        self.data = data
        # self.sess = sess
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0
        self.test = True
        '''
        B: batch

        SU: spare user feature
        DU: dense user feature

        T: items length
        DN: dense feature length
        SN: spare feature length

        BT: behavior length
        SBN: spare behavior feature length

        LN: ctr lable length
        '''
        # B*SU B*DU B*T*DN B*BT*SBN
        self.feature_key = ["user_spare_feature", "user_dense_feature", "item_spare_feature", "item_dense_feature",
                            "hist_spare_feature", "cur_mask"]
        # B*LN = B*T
        self.label_key = ["ctr_label"]
        print("epoch_size", self.epoch_size)

    def __iter__(self):
        return self

    def next(self):

        if self.i == self.epoch_size:
            raise StopIteration

        buffers = []

        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size,
                                                     len(self.data))]
        self.zero_count = 0
        for line in ts:
            # if self.test:
            #     print(line)
            #     self.test = False
            try:
                user_spare_profiles = line[1][0]
                user_dense_profiles = line[1][0]
                item_dense_feature = np.array(line[3])
                item_spare_feature = np.array(line[4])
                ctr_label = line[7]
                behavior_dense_feature = line[5]
                behavior_spare_feature = line[6]
                cur_feature = []

                buffers.append({
                    "user_spare_feature": np.array(user_spare_profiles).astype(int).tolist(),
                    "user_dense_feature": np.array(user_dense_profiles).astype(float).tolist(),
                    "item_spare_feature": item_dense_feature.astype(int).tolist(),  # 12 * N
                    "item_dense_feature": item_spare_feature.astype(float).tolist(),  # 12 * M
                    "hist_spare_feature": np.array(behavior_dense_feature).reshape(self.window_size, 2).astype(
                        int).tolist(),  # 12 * 2
                    "hist_dense_feature": np.array(behavior_spare_feature).reshape(self.window_size, 2).astype(
                        int).tolist(),  # 12 * 2
                    "ctr_label": np.array(ctr_label).astype(int).tolist()
                })
            except:
                print("error", line)
                continue

        self.i += 1
        # print("Have zeros list:", self.zero_count, " in batch size:", self.batch_size)

        return 0, buffers


class CBUSessionActorDataInput:
    def __init__(self, data, batch_size, window_size=50, item_seq_len=12):

        self.batch_size = batch_size
        self.window_size = window_size # 历史点击序列长度
        self.item_seq_len = item_seq_len # 候选序列长度
        self.data = data
        # self.sess = sess
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0
        self.test = True

        self.feature_key = ["user_spare_feature", "user_dense_feature", "item_spare_feature", "item_dense_feature", "hist_spare_feature", "cur_mask", "pre_item_click_spare_feature", "done"]
        # B*LN = B*T
        self.label_key = ["ctr_label", "conver_label", "final_score"]  # 点击率 转化率 精排得分
        self.buffer_key = ["state", "action", "reward", "next_state", "done"]
        print("epoch_size", self.epoch_size)
        self.totalNullNums = 0

    def __iter__(self):
        return self

    def next(self):

        if self.i == self.epoch_size:
            if self.totalNullNums > 0:
                print("The total batch have null line is:", self.totalNullNums)
            raise StopIteration

        buffers = []
        user_pvid_buffer = []

        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size,
                                                     len(self.data))]
        feature_len = (8 + 14 + 5 + 34 + 20 + 2 + 5 + 9 + 7 * self.window_size + 36 + 3 + 4)
        spare_feature_index = list(range(23)) + [27] + list(range(30,39)) + list(range(59,63))
        dense_feature_index = [i for i in range(66) if i not in spare_feature_index]
        # print("dense len:", len(dense_feature_index)) # 29
        # print("spare len:", len(spare_feature_index)) # 36
        self.zero_count = 0
        pre_click_item_feature = []
        totalNullNums = 0
        for index, line in enumerate(ts):
            # if self.test:
            #     print(line)
            #     self.test = False
            # try:
            user_spare_profiles = []
            user_dense_profiles = []
            cur_feature = []
            behavior_spare_feature = []
            cur_mask = []

            ctr_label = []
            conver_label = []
            score_label = []
            rewards = []

            user_id = line[0]
            if index == len(ts)-1 or user_id != ts[index+1][0]:
                done = 1
                pre_click_item_feature = []
            else:
                done = 0
            values = line[2].split(",")
            # print("value:", values)
            user_pvid_buffer.append(values[0])
            for i in range(1, self.item_seq_len + 1):
                if not user_spare_profiles:
                    user_spare_profiles = values[(i - 1) * feature_len + 1: (i - 1) * feature_len + 1 + 8]
                if not user_dense_profiles:
                    user_dense_profiles = values[(i - 1) * feature_len + 1 + 8: (i - 1) * feature_len + 1 + 8 + 14]
                
                cur_mask.append(values[i * feature_len - 3])

                cur_feature.append(values[(i - 1) * feature_len + 1 + 8 + 14 : (i - 1) * feature_len + 1 + 8 + 14 + 1 + 65])

                if not behavior_spare_feature:
                    behavior_spare_feature = values[(i - 1) * feature_len + 1 + 8 + 14 + 1 + 65 + 9: (i - 1) * feature_len + 1 + 8 + 14 + 1 + 65 + 9 + 7 * self.window_size]
                    final = []
                    for b_i in range(self.window_size):
                        final.append([behavior_spare_feature[b_i::50]])
                    behavior_spare_feature = final

                ctr_label.append(values[i * feature_len - 2]) # ctr_label
                conver_label.append(values[i * feature_len - 1]) # conver label
                score_label.append(values[i * feature_len - 4]) # model final score

                if values[i * feature_len - 2]:
                    if len(pre_click_item_feature) == 12:
                        pre_click_item_feature.pop(0)
                    pre_click_item_feature.append(np.array(cur_feature[-1])[spare_feature_index].tolist())


            cur_feature = np.array(cur_feature).astype(float)

            his_feature = np.array(behavior_spare_feature).reshape(self.window_size, 7).astype(float).astype(int)
            pre_feature = np.array(pre_click_item_feature).astype(float)[:, :7].astype(int)
            append_his_feature = np.zeros((62, 7))
            append_his_feature[:50, :] = his_feature
            append_his_feature[50:50+len(pre_feature), :] = pre_feature
            # print("line:", line)
            # print("user spare:", user_spare_profiles)
            # 处理User id为空的情况
            user_spare_feature = np.array(user_spare_profiles)
            user_dense_feature = np.array(user_dense_profiles)
            numsOfNull = np.sum((user_spare_feature == '').any(axis=-1))
            numsOfNullDense = np.sum((user_dense_feature == '').any(axis=-1))
            self.totalNullNums += numsOfNull
            self.totalNullNums += numsOfNullDense
            user_spare_feature = np.where(user_spare_feature == '', '0', user_spare_feature)
            user_dense_feature = np.where(user_dense_feature == '', '0', user_dense_feature)
            buffers.append({
                "user_spare_feature": user_spare_feature.astype(float).astype(int).tolist(),
                "user_dense_feature": user_dense_feature.astype(float).tolist(), 
                "item_spare_feature": cur_feature[:, spare_feature_index].astype(float).astype(int).tolist(),  # 12 * N
                "item_dense_feature": cur_feature[:, dense_feature_index].astype(float).tolist(), # 12 * M
                "hist_spare_feature": append_his_feature.astype(float).astype(int).tolist(),  # 50 * 7 ==> expand to 62 * 7
                "cur_mask": np.array(cur_mask).astype(float).astype(int).tolist(),
                "ctr_label": np.array(ctr_label).astype(float).astype(int).tolist(),
                "conver_label": np.array(conver_label).astype(float).tolist(),
                "score_label": np.array(score_label).astype(float).tolist(),
                "done": np.array(done).astype(float).astype(int).tolist()
            })

                
            # except Exception as e:
            #     print("error:", e)
            #     break

        self.i += 1
        # print("Have zeros list:", self.zero_count, " in batch size:", self.batch_size)
        # print("Buffer test:", buffers)
        return user_pvid_buffer, buffers

class CBUSessionKTimeActorDataInput:
    def __init__(self, data, batch_size, window_size=50, item_seq_len=12, k_seq=3):

        self.batch_size = batch_size
        self.window_size = window_size # 历史点击序列长度
        self.item_seq_len = item_seq_len # 候选序列长度
        self.data = data
        self.k_seq = k_seq # sequence length 用于序列offline RL推荐
        # self.sess = sess
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0
        self.test = False

        self.feature_key = ["user_spare_feature", "user_dense_feature", "item_spare_feature", "item_dense_feature", "hist_spare_feature", "cur_mask", "pre_item_click_spare_feature", "done"]
        # B*LN = B*T
        self.label_key = ["ctr_label", "conver_label", "final_score"]  # 点击率 转化率 精排得分
        self.buffer_key = ["state", "action", "reward", "next_state", "done"]
        print("epoch_size", self.epoch_size)
        self.totalNullNums = 0

    def __iter__(self):
        return self

    def next(self):

        def shift_array_onepeace(arr):
            arr[:self.k_seq-1] = arr[1:self.k_seq]
            arr[self.k_seq-1] *= 0
            return arr

        if self.i == self.epoch_size:
            if self.totalNullNums > 0:
                print("The total batch have null line is:", self.totalNullNums)
            raise StopIteration

        buffers = []

        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size,
                                                     len(self.data))]
        feature_len = (8 + 14 + 5 + 34 + 20 + 2 + 5 + 9 + 7 * self.window_size + 36 + 3 + 4)
        spare_feature_index = list(range(23)) + [27] + list(range(30,39)) + list(range(59,63))
        dense_feature_index = [i for i in range(66) if i not in spare_feature_index]
        # print("dense len:", len(dense_feature_index)) # 29
        # print("spare len:", len(spare_feature_index)) # 36
        self.zero_count = 0
        pre_click_item_feature = []
        totalNullNums = 0

        seq_ctr_label = np.zeros([self.k_seq, 12]).astype(float)
        seq_conver_label = np.zeros([self.k_seq, 12]).astype(float)
        seq_score_label = np.zeros([self.k_seq, 12]).astype(float)
        sequence_item = np.zeros([self.k_seq, 12, 66]).astype(float)
        seq_time = np.zeros([self.k_seq, 1]).astype(float)
        time_index = -1
        time_absolute = -1

        for index, line in enumerate(ts):   
            user_spare_profiles = []
            user_dense_profiles = []
            cur_feature = []
            behavior_spare_feature = []
            cur_mask = []

            ctr_label = []
            conver_label = []
            score_label = []
            
            rewards = []

            user_id = line[0]
            if index == len(ts)-1 or user_id != ts[index+1][0]:
                done = 1
                
            else:
                done = 0
            time_index += 1
            time_absolute += 1
            if time_index >= self.k_seq:
                time_index = self.k_seq - 1
                sequence_item = shift_array_onepeace(sequence_item)
                seq_ctr_label = shift_array_onepeace(seq_ctr_label)
                seq_conver_label = shift_array_onepeace(seq_conver_label)
                seq_score_label = shift_array_onepeace(seq_score_label)
                seq_time = shift_array_onepeace(seq_time)


            values = line[2].split(",")

            for i in range(1, self.item_seq_len + 1):
                if not user_spare_profiles:
                    user_spare_profiles = values[(i - 1) * feature_len + 1: (i - 1) * feature_len + 1 + 8]
                if not user_dense_profiles:
                    user_dense_profiles = values[(i - 1) * feature_len + 1 + 8: (i - 1) * feature_len + 1 + 8 + 14]
                
                cur_mask.append(values[i * feature_len - 3])

                cur_feature.append(values[(i - 1) * feature_len + 1 + 8 + 14 : (i - 1) * feature_len + 1 + 8 + 14 + 1 + 65])

                if not behavior_spare_feature:
                    behavior_spare_feature = values[(i - 1) * feature_len + 1 + 8 + 14 + 1 + 65 + 9: (i - 1) * feature_len + 1 + 8 + 14 + 1 + 65 + 9 + 7 * self.window_size]
                    final = []
                    for b_i in range(self.window_size):
                        final.append([behavior_spare_feature[b_i::50]])
                    behavior_spare_feature = final

                ctr_label.append(values[i * feature_len - 2]) # ctr_label
                conver_label.append(values[i * feature_len - 1]) # conver label
                score_label.append(values[i * feature_len - 4]) # model final score
            
            


            cur_feature = np.array(cur_feature).astype(float)
            sequence_item[time_index] = cur_feature
            cur_ctr = np.array(ctr_label).astype(float)
            seq_ctr_label[time_index] = cur_ctr
            cur_conver = np.array(conver_label).astype(float)
            seq_conver_label[time_index] = cur_conver
            cur_score = np.array(score_label).astype(float)
            seq_score_label[time_index] = cur_score

            seq_time[time_index] = time_absolute


            if self.test:
                print("user id:", user_id)
                print("cur feature:", cur_feature)
                print("ctr label:", ctr_label)
                print("time:", time_absolute)

                print("seq cur feature:", sequence_item)
                print("seq ctr label:", seq_ctr_label)
                print("seq time:", seq_time)

            if done:
                time_index = -1
                time_absolute = -1

                sequence_item = np.zeros([self.k_seq, 12, 66]).astype(float)
                seq_ctr_label = np.zeros([self.k_seq, 12]).astype(float)
                seq_conver_label = np.zeros([self.k_seq, 12]).astype(float)
                seq_score_label = np.zeros([self.k_seq, 12]).astype(float)
                seq_time = np.zeros([self.k_seq, 1]).astype(float)

            his_feature = np.array(behavior_spare_feature).reshape(self.window_size, 7).astype(float).astype(int)

            # 处理User id为空的情况
            user_spare_feature = np.array(user_spare_profiles)
            user_dense_feature = np.array(user_dense_profiles)
            numsOfNull = np.sum((user_spare_feature == '').any(axis=-1))
            numsOfNullDense = np.sum((user_dense_feature == '').any(axis=-1))
            self.totalNullNums += numsOfNull
            self.totalNullNums += numsOfNullDense
            user_spare_feature = np.where(user_spare_feature == '', '0', user_spare_feature)
            user_dense_feature = np.where(user_dense_feature == '', '0', user_dense_feature)
            buffers.append({
                "user_spare_feature": user_spare_feature.astype(float).astype(int).tolist(),
                "user_dense_feature": user_dense_feature.astype(float).tolist(), 
                "item_spare_feature": sequence_item[:, :, spare_feature_index].astype(float).astype(int).tolist(),  # seq * 12 * N
                "item_dense_feature": sequence_item[:, :, dense_feature_index].astype(float).tolist(), # seq * 12 * M
                "hist_spare_feature": his_feature.astype(float).astype(int).tolist(),  # 50 * 7 ==> expand to 62 * 7
                "cur_mask": np.array(cur_mask).astype(float).astype(int).tolist(),
                "ctr_label": np.array(seq_ctr_label).astype(float).astype(int).tolist(),
                "conver_label": np.array(seq_conver_label).astype(float).tolist(),
                "single_ctr_label": np.array(ctr_label).astype(float).astype(int).tolist(),
                "single_score_label":  np.array(score_label).astype(float).astype(int).tolist(),
                "score_label": np.array(seq_score_label).astype(float).tolist(),
                "done": np.array(done).astype(float).astype(int).tolist(),
                "time": np.array(seq_time).astype(float).astype(int).tolist(),
            })

                
            # except Exception as e:
            #     print("error:", e)
            #     break

        self.i += 1
        self.test = False
        # print("Have zeros list:", self.zero_count, " in batch size:", self.batch_size)
        # print("Buffer test:", buffers)
        return 0, buffers


class AvitoSessionKTimeActorDataInput:
    def __init__(self, data, batch_size, window_size=12, item_seq_len=12, k_seq=3):

        self.batch_size = batch_size
        self.window_size = window_size  # 历史点击序列长度
        self.item_seq_len = item_seq_len  # 候选序列长度
        self.data = data
        self.k_seq = k_seq  # sequence length 用于序列offline RL推荐
        # self.sess = sess
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0
        self.test = False

        self.feature_key = ["user_spare_feature", "user_dense_feature", "item_spare_feature", "item_dense_feature",
                            "hist_spare_feature", "cur_mask", "pre_item_click_spare_feature", "done"]
        # B*LN = B*T
        self.label_key = ["ctr_label", "conver_label", "final_score"]  # 点击率 转化率 精排得分
        self.buffer_key = ["state", "action", "reward", "next_state", "done"]
        print("epoch_size", self.epoch_size)
        self.totalNullNums = 0

    def __iter__(self):
        return self

    def next(self):

        def shift_array_onepeace(arr):
            arr[:self.k_seq - 1] = arr[1:self.k_seq]
            arr[self.k_seq - 1] *= 0
            return arr

        if self.i == self.epoch_size:
            if self.totalNullNums > 0:
                print("The total batch have null line is:", self.totalNullNums)
            raise StopIteration

        buffers = []

        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size,
                                                     len(self.data))]
        feature_len = (8 + 14 + 5 + 34 + 20 + 2 + 5 + 9 + 7 * self.window_size + 36 + 3 + 4)
        spare_feature_index = list(range(23)) + [27] + list(range(30, 39)) + list(range(59, 63))
        dense_feature_index = [i for i in range(66) if i not in spare_feature_index]
        # print("dense len:", len(dense_feature_index)) # 29
        # print("spare len:", len(spare_feature_index)) # 36
        self.zero_count = 0
        pre_click_item_feature = []
        totalNullNums = 0

        seq_ctr_label = np.zeros([self.k_seq, 12]).astype(float)
        seq_conver_label = np.zeros([self.k_seq, 12]).astype(float)
        seq_score_label = np.zeros([self.k_seq, 12]).astype(float)
        sequence_item_dense = np.zeros([self.k_seq, 12, 2]).astype(float)
        sequence_item_spare = np.zeros([self.k_seq, 12, 2]).astype(float)
        seq_time = np.zeros([self.k_seq, 1]).astype(float)
        time_index = -1
        time_absolute = -1

        for index, line in enumerate(ts):
            user_spare_profiles = []
            user_dense_profiles = []
            cur_feature = []
            behavior_spare_feature = []
            cur_mask = []

            ctr_label = []
            conver_label = []
            score_label = []

            rewards = []

            user_id = line[0]
            if index == len(ts) - 1 or user_id != ts[index + 1][0]:
                done = 1

            else:
                done = 0
            time_index += 1
            time_absolute += 1
            if time_index >= self.k_seq:
                time_index = self.k_seq - 1
                sequence_item_dense = shift_array_onepeace(sequence_item_dense)
                sequence_item_spare = shift_array_onepeace(sequence_item_spare)
                seq_ctr_label = shift_array_onepeace(seq_ctr_label)
                seq_conver_label = shift_array_onepeace(seq_conver_label)
                seq_score_label = shift_array_onepeace(seq_score_label)
                seq_time = shift_array_onepeace(seq_time)

            user_spare_profiles = line[1][0]
            user_dense_profiles = line[1][0]
            item_dense_feature = np.array(line[3])
            item_spare_feature = np.array(line[4])
            ctr_label = line[7]
            score_label = [0] * self.item_seq_len
            behavior_dense_feature = line[5]
            behavior_spare_feature = line[6]


            cur_feature = np.array(cur_feature).astype(float)
            sequence_item_dense[time_index] = item_dense_feature
            sequence_item_spare[time_index] = item_spare_feature
            cur_ctr = np.array(ctr_label).astype(float)
            seq_ctr_label[time_index] = cur_ctr
            # cur_conver = np.array(conver_label).astype(float)
            # seq_conver_label[time_index] = cur_conver
            # cur_score = np.array(score_label).astype(float)
            # seq_score_label[time_index] = cur_score

            seq_time[time_index] = time_absolute

            if self.test:
                print("user id:", user_id)
                print("cur feature:", cur_feature)
                print("ctr label:", ctr_label)
                print("time:", time_absolute)

                print("seq cur feature:", sequence_item_dense)
                print("seq ctr label:", seq_ctr_label)
                print("seq time:", seq_time)

            if done:
                time_index = -1
                time_absolute = -1
                sequence_item_dense = np.zeros([self.k_seq, 12, 2]).astype(float)
                sequence_item_spare = np.zeros([self.k_seq, 12, 2]).astype(float)
                seq_ctr_label = np.zeros([self.k_seq, 12]).astype(float)
                seq_conver_label = np.zeros([self.k_seq, 12]).astype(float)
                seq_score_label = np.zeros([self.k_seq, 12]).astype(float)
                seq_time = np.zeros([self.k_seq, 1]).astype(float)

            his_feature = np.array(behavior_spare_feature).reshape(self.window_size, 2).astype(float).astype(int)

            # 处理User id为空的情况
            user_spare_feature = np.array(user_spare_profiles)
            user_dense_feature = np.array(user_dense_profiles)
            numsOfNull = 0 #np.sum((user_spare_feature == '').any(axis=-1))
            numsOfNullDense = 0 #np.sum((user_dense_feature == '').any(axis=-1))
            self.totalNullNums += numsOfNull
            self.totalNullNums += numsOfNullDense
            user_spare_feature = np.where(user_spare_feature == '', '0', user_spare_feature)
            user_dense_feature = np.where(user_dense_feature == '', '0', user_dense_feature)
            buffers.append({
                "user_spare_feature": user_spare_feature.astype(float).astype(int).tolist(),
                "user_dense_feature": user_dense_feature.astype(float).tolist(),
                "item_spare_feature": sequence_item_spare.astype(float).astype(int).tolist(),
                # seq * 12 * N
                "item_dense_feature": sequence_item_dense.astype(float).tolist(),  # seq * 12 * M
                "hist_spare_feature": his_feature.astype(float).astype(int).tolist(),  # 50 * 2 ==> expand to 62 * 2
                "cur_mask": np.array(cur_mask).astype(float).astype(int).tolist(),
                "ctr_label": np.array(seq_ctr_label).astype(float).astype(int).tolist(),
                "conver_label": np.array(seq_conver_label).astype(float).tolist(),
                "single_ctr_label": np.array(ctr_label).astype(float).astype(int).tolist(),
                "single_score_label": np.array(score_label).astype(float).astype(int).tolist(),
                "score_label": np.array(seq_score_label).astype(float).tolist(),
                "done": np.array(done).astype(float).astype(int).tolist(),
                "time": np.array(seq_time).astype(float).astype(int).tolist(),
            })

            # except Exception as e:
            #     print("error:", e)
            #     break

        self.i += 1
        self.test = False
        # print("Have zeros list:", self.zero_count, " in batch size:", self.batch_size)
        # print("Buffer test:", buffers)
        return 0, buffers

if __name__ == "__main__":

    tf.app.flags.DEFINE_string("tables", "", "tables info")

    FLAGS = tf.app.flags.FLAGS

    print("tables:" + FLAGS.tables)
    tables = FLAGS.tables
    tables = FLAGS.tables.split(",")
    print("split tables", tables)

    dataset_reader = tf.python_io.TableReader(tables[1])
    total_records_num = dataset_reader.get_row_count()
    dataset_set = dataset_reader.read(100)
    dataset_reader.close()

    buffer_name = ["states", "actions", "rewards", "next_states", "dones"]


    # for _, uij in DataInput(dataset_set, 12):
    #     print("="*50)
    #     print("Buffers:")
    #     print(uij)
    #     for j in range(len(uij)):
    #         for i in buffer_name:
    #             print(i, ":")
    #             print(uij[j][i])
    #     break
    buffer_name = ["user_spare_feature", "user_dense_feature", "item_spare_feature", "item_dense_feature", "hist_spare_feature"] + ["ctr_label", "conver_label", "final_score"] + ["done"]
    for _, uij in CBUSessionKTimeActorDataInput(dataset_set, 2):
        print("="*50)
        batch = get_aggregated_batch4cbu_session_k_seq(uij)
        print(batch)
