import gym
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils import *
from tfmodel import *

env = gym.make("CartPole-v0")
sess = tf.Session()
model = PolicyModel(env, num_actions=2, input_shape=4, sess=sess, learning_rate=0.001, batch_size=10)
gg = model.train()