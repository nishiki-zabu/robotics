#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import deque
import os
import numpy as np
import tensorflow as tf
from numpy.random import *
from tensorflow.contrib.learn.python.learn.datasets import mnist
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec

class DQNAgent_RandomWalk:
    """
    Multi Layer Perceptron with Experience Replay
    """

    def __init__(self, enable_actions, environment_name):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.environment_name = environment_name
        self.enable_actions = enable_actions
        self.n_actions = len(self.enable_actions)
        self.minibatch_size = 32
        self.replay_memory_size = 100000
        self.learning_rate = 0.001
        self.discount_factor = 0.9
        self.exploration = 0.1
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.model_name = "{}.ckpt".format(self.environment_name)
        # replay memory
        self.D = deque(maxlen=self.replay_memory_size)
        # model
        self.init_model()

    def init_model(self):
        # 重みを標準偏差0.1の正規分布で初期化
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)
        # バイアスを標準偏差0.1の正規分布で初期化
        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)
        # 畳み込み層の作成
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')
        # プーリング層の作成
        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print"init layer"

        """
        ※計算式: (inputのx(y)軸の要素数-filter(weight)のx(y)軸の要素数)/strideの要素数 + 1
        1. 入力イメージ => 28x28x3
        2. conv1 => (28-5)/1 + 1 = 24
        3. pool1 => (24-2)/2 + 1 = 12
        4. conv2 => (12-5)/1 + 1 = 8
        5. pool2 => ( 8-2)/2 + 1 = 4 =====> よって 4x4x64 のテンソルが得られる
        ※計算式: (inputのx(y)軸の要素数)/strideの要素数
        1. 入力イメージ => 28x28x3
        2. conv1 => (28)/1 = 28
        3. pool1 => (28)/2 = 14
        4. conv2 => (14)/1 = 14
        5. pool2 => (14)/2 =  7 =====> よって 7x7x64 のテンソルが得られる
        """

        # input layer (480x480)
        self.x = tf.placeholder(tf.float32, [None, 480, 480])

        # 入力を480x480x1に変形
        x_images = tf.reshape(self.x, [-1, 480, 480, 1])

        # 畳み込み層1の作成
        W_conv1 = weight_variable([24, 24, 1, 16])
        b_conv1 = bias_variable([16])
        h_conv1 = tf.nn.relu(conv2d(x_images, W_conv1) + b_conv1)

        # プーリング層1の作成
        h_pool1 = max_pool_2x2(h_conv1)

        # 畳み込み層2の作成
        W_conv2 = weight_variable([12, 12, 16, 32])
        b_conv2 = bias_variable([32])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        # プーリング層2の作成
        h_pool2 = max_pool_2x2(h_conv2)

        # 畳み込み層3の作成
        W_conv3 = weight_variable([6, 6, 32, 64])
        b_conv3 = bias_variable([64])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

        h_out = tf.reshape(h_conv3,[-1,15*15*64])
        print"fully flatten"

        # output layer (n_actions)
        W_out = tf.Variable(tf.truncated_normal([15*15*64, self.n_actions], stddev=0.01))
        b_out = tf.Variable(tf.zeros([self.n_actions]))
        self.y = tf.matmul(h_out, W_out) + b_out
        print"output layer"

        # loss function
        self.y_ = tf.placeholder(tf.float32, [None, self.n_actions])
        self.loss = tf.reduce_mean(tf.square(self.y_ - self.y))
        print"loss function"

        # train operation
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.training = optimizer.minimize(self.loss)
        print"train operation"

        # saver
        self.saver = tf.train.Saver()
        print"saver"

        # session
        self.sess = tf.Session()
        print"sess"

        self.sess.run(tf.global_variables_initializer())
        print"finish init layer"

    def Q_values(self, state):
        # Q(state, action) of all actions
        return self.sess.run(self.y, feed_dict={self.x: [state]})[0]

    # epsilon greedy
    def select_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            # random
            return np.random.choice(self.enable_actions)
        else:
            # max_action Q(state, action)
            return self.enable_actions[np.argmax(self.Q_values(state))]

    def store_experience(self, state, action, reward, state_1):
        self.D.append((state, action, reward, state_1))

    def experience_replay(self):
        state_minibatch = []
        y_minibatch = []

        # sample random minibatch
        minibatch_size = min(len(self.D), self.minibatch_size)
        minibatch_indexes = np.random.randint(0, len(self.D), minibatch_size)

        for j in minibatch_indexes:
            state_j, action_j, reward_j, state_j_1 = self.D[j]
            action_j_index = self.enable_actions.index(action_j)
            y_j = self.Q_values(state_j)

            # reward_j+gamma*max_action'Q(state', action')
            y_j[action_j_index] = reward_j+self.discount_factor*np.max(self.Q_values(state_j_1))
            state_minibatch.append(state_j)
            y_minibatch.append(y_j)

        # training
        self.sess.run(self.training, feed_dict={self.x: state_minibatch, self.y_: y_minibatch})

    def load_model(self, model_path=None):
        if model_path:
            # load from model_path
            self.saver.restore(self.sess, model_path)
        else:
            # load from checkpoint
            checkpoint = tf.train.get_checkpoint_state(self.model_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

    def save_model(self):
        self.saver.save(self.sess, os.path.join(self.model_dir, self.model_name))
