import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils import *
""" Some pointers from https://github.com/carpedm20/DCGAN-tensorflow """

def deep_q_net(input_, num_actions):
    h0 = conv2d(input_, 16, 5, 5, 2, 2, name='h0_conv')
    h1 = conv2d(h0, 32, 5, 5, 2, 2, name='h1_conv')
    h2 = fc_relu(tf.reshape(h1, shape=[-1, 3200]), 256, name='h2_relu')
    action_logits = fc_linear(h2, num_actions, name='action_logits')
    value_function = fc_linear(h2, 1, name='values')
    return tf.nn.softmax(action_logits), action_logits, value_function

class PolicyModel(object):
    def __init__(self, env, num_actions, input_shape, sess, learning_rate, batch_size):
        self.env = env
        self.num_actions = num_actions
        self.input_shape = input_shape
        self.batch_size = 3
        self.sess = sess
        self.learning_rate = learning_rate
        self.init_tf_variables()

    def init_tf_variables(self):
        width, height, depth = self.input_shape
        self.input_ = tf.placeholder(dtype=tf.float32, shape=[None, width, height, depth])
        self.actions = tf.placeholder(tf.int32, (None,), name="actions")
        self.discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")
        action_probs, action_logits, value_function = deep_q_net(self.input_, self.num_actions)
        self.action_prob = action_probs
        self.action_logits = action_logits #stored for sampling an action in an episode
        self.action_index = tf.multinomial(action_logits - tf.reduce_max(action_logits, 1, keep_dims=True), 1)

        self.loss = tf.reduce_mean((self.discounted_rewards) * \
                tf.nn.sparse_softmax_cross_entropy_with_logits(action_logits, self.actions))
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)
        self.sess.run((tf.initialize_all_variables()))

    def run_episode(self):
        observations = [prepro_grey(self.env.reset(), (self.input_shape[0], self.input_shape[1]))]
        width, height, depth = self.input_shape
        rewards = []
        actions = []
        frames = []
        current_framestack = np.zeros(self.input_shape) #depending on how many frames we want to keep in history (depth of input)
        current_framestack[:,:,0] = observations[0]
        current_framestack = np.roll(current_framestack, 1, 2)
        done = False

        while not done:
            if self.episode_number % 30 == 0:
                self.env.render()
            # loosely following https://github.com/karpathy/tf-agent/blob/master/policy_gradient.py
            action = self.sess.run(self.action_index, {self.input_ : current_framestack.reshape(1, width, height, depth)})
            action = action[0][0]
            # (Pong): 1 = no op, 2 = up, 3 = down. in nn softmax, 0 = no op, 1 = up, 2 = down
            observation, reward, done, info = self.env.step(action+1)
            observation = prepro_grey(observation, (self.input_shape[0], self.input_shape[1]))
            # np.roll rolls the array so that the previous last one is now the first one.
            current_framestack = np.roll(current_framestack, 1, 2)
            #overwrite with the new frame
            current_framestack[:,:,0] = observation
            actions.append(action)
            rewards.append(reward)
            frames.append(current_framestack.copy())

        discounted_rewards = discount_rewards(np.array(rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        return actions, discounted_rewards.tolist(), rewards, frames


    def train(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        self.episode_number = 0
        while self.episode_number < 500:
            actions = []
            discounted_rewards = []
            rewards = []
            frames = []

            for i in range(self.batch_size):
                self.episode_number += 1
                ep_actions, ep_discounted, ep_rewards, ep_frames = self.run_episode()
                actions += ep_actions
                discounted_rewards += ep_discounted
                rewards += ep_rewards
                frames += ep_frames

            frames = np.stack(frames)

            _, loss = self.sess.run([self.train_op, self.loss],
                            feed_dict = {
                                self.input_ : frames,
                                self.actions : actions,
                                self.discounted_rewards : discounted_rewards
                            }
                        )

            print('Rewards for batch: %f' % (np.sum(rewards)))
            print('loss: %f' % (loss))