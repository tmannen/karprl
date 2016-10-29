""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle
import gym
import matplotlib
import matplotlib.pyplot as plt
from tfmodel import *

env = gym.make("Pong-v0")
sess = tf.Session()
model = PolicyModel(env, num_actions=3, input_shape=(40, 40, 4), sess=sess, learning_rate=0.001, batch_size=3)
gg = model.train()