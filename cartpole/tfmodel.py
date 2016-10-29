import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gym
from utils import *
""" Some pointers from https://github.com/carpedm20/DCGAN-tensorflow """

def simple_network(input_, num_actions):
    h1 = fc_relu(input_, 25, name='h1_relu')
    h2 = fc_relu(h1, 25, name='h2_relu')
    action_logits = fc_linear(h2, num_actions, name='action_logits')
    val_h1 = fc_linear(input_, 25, name='val_h1')
    value_function = fc_linear(val_h1, 1, name='value_function')
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
        self.input_ = tf.placeholder(dtype=tf.float32, shape=[None, self.input_shape])
        self.actions = tf.placeholder(tf.int32, (None,), name="actions")
        self.rewards = tf.placeholder(tf.float32, (None,), name="rewards")
        action_prob, action_logits, value_function = simple_network(self.input_, self.num_actions)
        self.action_prob = action_prob
        self.action_logits = action_logits #stored for sampling an action in an episode
        self.action_index = tf.multinomial(action_logits - tf.reduce_max(action_logits, 1, keep_dims=True), 1)

        self.pg_loss = tf.reduce_mean((self.rewards - value_function) * \
                tf.nn.sparse_softmax_cross_entropy_with_logits(action_logits, self.actions))
        self.value_loss = 0.5*tf.reduce_mean(tf.square(self.rewards - value_function))
        self.loss = self.pg_loss + self.value_loss

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)
        self.sess.run((tf.initialize_all_variables()))

    def run_episode(self):
        observation = self.env.reset()
        observations = []
        rewards = []
        actions = []
        done = False

        while not done:
            if self.episode_number % 1000 == 0:
                self.env.render()
            # 0 or 1 for cartpole
            action_prob = self.sess.run(self.action_prob, {self.input_ : observation.reshape((1, -1))})
            action = 0 if np.random.uniform() < action_prob[0][0] else 1
            old_observation = observation # if last step was good, reward that
            observation, reward, done, info = self.env.step(action)
            actions.append(action)
            rewards.append(reward)
            observations.append(old_observation)

        return actions, rewards, observations


    def train(self):
        self.episode_number = 0
        while self.episode_number < 100000:
            actions = []
            rewards = []
            observations = []

            for i in range(self.batch_size):
                self.episode_number += 1
                ep_actions, ep_rewards, ep_frames = self.run_episode()
                actions += ep_actions
                rewards += ep_rewards
                observations += ep_frames
                if len(rewards) > 600:
                    print("happening")
                    break

            if self.episode_number % 500 == 0:
                print(np.sum(rewards))
            observations = np.stack(observations)
            discounted_rewards = discount_rewards(rewards)

            _, loss = self.sess.run([self.train_op, self.loss],
                            feed_dict = {
                                self.input_ : observations,
                                self.actions : actions,
                                self.rewards : discounted_rewards
                            }
                        )

            #print('Rewards for batch: %f' % (np.sum(rewards)))
            #print('loss: %f' % (loss))