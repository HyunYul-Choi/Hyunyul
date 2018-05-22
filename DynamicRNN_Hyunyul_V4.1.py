# -*- coding: utf-8 -*-
"""
Created on Mon May 21 19:07:37 2018

@author: HyunYul.Choi
"""
#from scipy import stats
from __future__ import print_function, division, absolute_import
#import win32api
from time import time
import os
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib import rnn
import random
import keras
import keras.backend as K
import numpy as np
import pandas as pd
import pickle
import warnings
import matplotlib
import matplotlib.pyplot as plt
import re
import time
import collections
import scipy.sparse as sp
os.chdir("D:/Dropbox/DMQA_최현율/★ 삼성SDS 스타크래프트/DMQA_starcraft-master/preprocessing")
from utils import custom_resize

#Set GPU memory usage threshold
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings
# file path
path_dir = "D:/data/starcraft/trainingData_v0/"
directory = os.listdir(path_dir)

# Win & Lose Data listup
win_list=[]
lose_list=[]
for i in range(0,len(directory)):
    if directory[i].split("_")[0] == '1':
        win_list.append(directory[i])
    elif directory[i].split("_")[0] == '0':
        lose_list.append(directory[i])

with open('D:/data/starcraft/trainingData_v0/0_PWTL_박성균_72_b799996378ccea7369e02749015b5839dd03bf6d_256.pkl', 'rb') as f:
    replay = pickle.load(f)
    print(">>> The replay file is a dictionary with the following keys: '{}', '{}'.".format(*replay.keys()))
    
replay_info = replay.get('replay_info')
for k, v in replay_info.items():
    print ("{}: {}".format(k, v))

replay_data = replay.get('replay_data')
print(">>> Number of keys (#. of samples): {}".format(replay_data.keys().__len__()))
len(replay_data)

# Adjusted Shape
#sample_index=0
for sample_index in range(0,len(replay_data)):
    sample, sample_info = replay_data.get(sample_index)
    sample = sample.toarray()
    print(">>> Shape of sample in 2D: {}".format(sample.shape))
    original_size = int(np.sqrt(sample.shape[0]))
    num_channels = sample.shape[-1]
    sample = sample.reshape((original_size, original_size, num_channels))
    print(">>> A single sample is a 3D numpy array of shape {} (original)".format(sample.shape))
    output_size = int(original_size / 2)
    sample = custom_resize(sample, output_size=output_size)
    sequence_length = len(replay_data.keys())
    samples = [s for s, _ in replay_data.values()]
    samples = [s.toarray() for s in samples]
    samples = [s.reshape((original_size, original_size, num_channels)) for s in samples]
    samples = [custom_resize(s, output_size) for s in samples]
    samples = np.stack(samples, axis=0)
    samples = samples.astype(np.float32)
    samples = samples.reshape((1, ) + samples.shape)

# RNNs will require 5-dimensional inputs of shape (B, T, H, W, C)
print(">>> A training observation is a 5D numpy array of shape {}".format(samples.shape))

# ========================Convolutional========================================
tf.set_random_seed(777)
# Hyperparameters
learning_rate = 0.01
training_epochs = 100
batch_size = 1

keep_prob = tf.placeholder(tf.float32) # dropout rate 0.7~0.5 
# input placeholders
X = tf.placeholder(tf.float32, shape=[None, 136])   #None= N개의 이미지, 136개 피클 파일
X_img= tf.reshape(X, [-1,128,128, 25])  #색깔 25개 (유닛 채널25)
Y = tf.placeholder(tf.float32, shape=[None, 10])
#★ tf.reset_default_graph()  # 배치 걸려있는 부분 전체 초기화(Hyperparameter 조정시 항시ON/OFF)★
# L1 ImgIn shape
W1 = tf.Variable(tf.random_normal([128, 128, 25, 32], stddev=0.01))
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
'''
Tensor("Conv2D:0", shape=(?, 64, 64, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 64, 64, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 64, 64, 32), dtype=float32)
Tensor("dropout/mul:0", shape=(?, 64, 64, 32), dtype=float32)
'''
# L2 ImgIn shape
W2 = tf.Variable(tf.random_normal([128, 128, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
'''
Tensor("Conv2D_1:0", shape=(?, 32, 32, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 32, 32, 64), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 32, 32, 64), dtype=float32)
Tensor("dropout_1/mul:0", shape=(?, 32, 32, 64), dtype=float32)
'''
# L3 ImgIn shape
W3 = tf.Variable(tf.random_normal([128, 128, 64, 128], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                    1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L3_flat = tf.reshape(L3, [-1, 128 * 16 * 16])
'''
Tensor("Conv2D_2:0", shape=(?, 16, 16, 128), dtype=float32)
Tensor("Relu_2:0", shape=(?, 16, 16, 128), dtype=float32)
Tensor("MaxPool_2:0", shape=(?, 16, 16, 128), dtype=float32)
Tensor("dropout_2/mul:0", shape=(?, 16, 16, 128), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 32768), dtype=float32)
'''
# L4 FC 16x16x128 inputs -> 100 outputs
W4 = tf.get_variable("W4", shape=[16 * 16 * 128, 100],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([100]))
L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
'''
Tensor("Relu_3:0", shape=(?, 100), dtype=float32)
Tensor("dropout_3/mul:0", shape=(?, 100), dtype=float32)
'''
keep_prob = tf.placeholder(tf.float32) #Dropout 되지 않을 확률을 저장할 placeholder 만들어 놓음.
# L5 Final FC 100 inputs -> 10 outputs
W5 = tf.get_variable("W5", shape=[100, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L4, W5) + b5
'''
Tensor("add_1:0", shape=(?, 10), dtype=float32)
'''
# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize of session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train Model
#print('Learning started. It takes sometime.')
#for epoch in range(training_epochs):
#    avg_cost = 0
#    total_batch = int(mnist.train.num_examples / batch_size)  # <- batch 들어가는 부분만 변경하면 됨.
#
#    for i in range(total_batch):
#        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
#        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
#        avg_cost += c / total_batch
#
#    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
#
#print('Learning Finished!')
sess.close()
print(cost)
print(optimizer)
print(logits)

# Concatenate
replay = np.concatenate((win_list, lose_list), axis=0)
init_shape = (replay[0].shape)
#print(">>> replay shape: {}".format(init_shape))
  
# Data_y
data_y = [[1,0]]*win_list.shape[0] + [[0,1]]*lose_list.shape[0]
data_y = np.array(data_y)
print(">>>data_y input_shape: {}".format(data_y.shape))

# ===================== Dynamic RNN ==========================================
sentence = replay
char_set = list(set(sentence))  # index → Char
char_dic = {c: i for i, c in enumerate(char_set)}  # Char → index

# Hyperparameter (Auto)
data_dim = len(char_set)     # RNN input size (One hot size)
hidden_size = len(char_set)  # RNN output size(자유롭게 출력할 정해)
num_classes = len(char_set)  # Final output size (RNN or softmax, etc.)
sequence_length = len(sentence) - 1  # Num. of LSTM rollings
print(sequence_length)
batch_size = 1  # one sample data, one batch
learning_rate = 0.01
#tf.reset_default_graph()  # 배치 걸려있는 부분 전체 초기화(Hyperparameter 조정시 항시 ON/OFF)★

dataX = []
dataY = []

for i in range(0, len(sentence) - sequence_length):  #윈도우를 움직이면서 Slicing 이용해서 스트링 뽑아냄.
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1: i + sequence_length + 1]
    print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]  # x str to index
    y = [char_dic[c] for c in y_str]  # y str to index

    dataX.append(x)
    dataY.append(y)
    
batch_size = len(char_dic)  #데이터 전체의 길이 넣어 줌


keep_prob = tf.placeholder(tf.float32) # dropout rate 0.7~0.5 
X = tf.placeholder(tf.float32, [batch_size, None, sequence_length])  # X data
'''
<tf.Tensor 'Placeholder_4:0' shape=(136, ?, 135) dtype=float32>
'''
#Y = tf.placeholder(tf.float32, [batch_size, None, sequence_length])  # Y label
Y = tf.placeholder(tf.int32, [batch_size, None, sequence_length])  # Y label
'''
<tf.Tensor 'Placeholder_5:0' shape=(136, ?, 135) dtype=float32>
'''

# Real-hyperparameter
num_epochs = 100
total_series_length = 102400
truncated_backprop_length = 128  
state_size = 1                   # State_size=1 로 무조건 시작해야 함.
num_classes = 2 
echo_step = 24
batch_size = 128
num_batches = total_series_length//batch_size//truncated_backprop_length #나누기 연산 후 소수점 이하의 수를 버리고, 정수 부분의 수만 구함
print(num_batches)
'''
num_batches = 6
'''
n_layers = 3                     # rnn_tuple_state = 3 fixed.

# One-hot encoding
#X_one_hot = tf.one_hot(X, num_classes) # DataType float32 not in list of allowed values:int32
X_one_hot = tf.one_hot(x, num_classes) # DataType float32 not in list of allowed values:int32
print(X_one_hot)  # check out the shape
'''
Tensor("one_hot_4:0", shape=(135, 2), dtype=float32)
'''
# Flatten the data
X_for_softmax = tf.reshape(X_one_hot, [-1, hidden_size])

# softmax layer (rnn_hidden_size -> num_classes)
softmax_w = tf.get_variable("softmax_w", [hidden_size, num_classes])
softmax_b = tf.get_variable("softmax_b", [num_classes])
outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b  #softmax의 아웃풋

# Make a lstm cell with hidden_size (each unit output vector size)
def lstm_cell():
    cell = tf.contrib.rnn.Conv2DLSTMCell(hidden_size, state_is_tuple=True)
    return cell
# MultiRNNCell (Deep하게)
multi_cells = tf.nn.rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)

# Outputs: unfolding size x hidden size, state = hidden size
outputs, _states = tf.nn.dynamic_rnn(multi_cells, X_one_hot, dtype=tf.float32)

# Fully Connected layer (Softmax 붙이기)
X_for_fc = tf.reshape(outputs, [-1, hidden_size])    #X_for_softmax : 입력은 RNN의 Output
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

# expended the data (revive the batches) (Softmax 붙여주기-이거에 맞는 Weight만 해주면 됨)
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes]) #softmax에서 나온 output을 펼쳐라
weights = tf.ones([batch_size, sequence_length]) # All Weights are 1 .

# Computing the sequence COST/loss  주의: Activation을 거치지 않은, 상기의 output을 logits에 넣어야 한다.
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)             # Mean all sequence loss
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

sentence = replay_data
sentence_idx = [char_set[c] for c in sentence]  # char to index
x_data = [sentence_idx[:-1]]  # X data sample (0 ~ n-1) hello: hell
y_data = [sentence_idx[1:]]   # Y label sample (1 ~ n) hello: ello

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
for i in range(3000):
    l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
    result = sess.run(prediction, feed_dict={X: x_data})

    # print char using dic
    result_str = [char_set[c] for c in np.squeeze(result)]
    print(i, "loss:", l, "Prediction:", ''.join(result_str))

print(result)
sess.close()