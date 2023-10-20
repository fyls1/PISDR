#coding=utf-8
import numpy as np
import time
import datetime
import tensorflow as tf
import time

def get_aggregated_batch4kuaishou(batch_data):
    '''
    batch
    '''
    buffers = batch_data

    buffers_used_keys = ["states", "next_states", "actions", "rewards", "dones"]
    buffers_used_keys_size = [1044, 1044, 1, 7, 1]

    batch_size = len(buffers)

    result = {}

    for index, item in enumerate(buffers_used_keys):
        res_item = np.array([buffers[i][item] for i in range(len(buffers))]).reshape(batch_size, buffers_used_keys_size[index])
        # print(item, " shape is:", res_item.shape)
        result[item] = res_item

    return result

class DataInput:
    def __init__(self, data, batch_size):

        self.batch_size = batch_size
        self.state_dim = 1044  # user feature: 24  cancadite videos: 150 * 6(feature length)  history video 20 * 6(feature length)
        self.reward_dim = 7
        self.data = data
        # self.sess = sess
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0
        self.test = True
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
            # if self.test:
            #     print(line)
            #     self.test = False
            try:
                # print("line length:", len(line))
                states = line[0].split("+")
                # print("len value:", len(states))
                # print("value:", states)
                next_states = line[1].split("+")
                actions = line[2]
                rewards = line[3].split("+")
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


if __name__ == "__main__":

    tf.app.flags.DEFINE_string("tables", "", "tables info")

    FLAGS = tf.app.flags.FLAGS

    print("tables:" + FLAGS.tables)
    tables = FLAGS.tables
    # tables = FLAGS.tables.split(",")
    print("split tables", tables)

    dataset_reader = tf.python_io.TableReader(tables)
    total_records_num = dataset_reader.get_row_count()
    dataset_set = dataset_reader.read(10)
    dataset_reader.close()

    buffer_name = ["states", "next_states", "actions", "rewards", "dones"]


    for _, uij in DataInput(dataset_set, 2):
        print("="*50)
        print("Buffers:")
        print(uij)
        for i in buffer_name:
            print(i, ":")
            print(uij[0][i])
