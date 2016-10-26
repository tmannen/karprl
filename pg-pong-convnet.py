""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle
import gym
import matplotlib
import matplotlib.pyplot as plt
from tfmodel import *

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 2 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid
if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H)
  
grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory


def prepro(I):
  plt.imshow(I)
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def prepro_grey(image):
  """ crop and downsample, and preprocess the image as in the Deep Q learning paper: stack 4 previous frames depth wise """
  #http://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
  def crop_and_rgb2gray(I):
    I = I[35:195] # crop
    I = I[::2,::2,:] # downsample by factor of 2
    return np.dot(I, [0.299, 0.587, 0.114])

  greyed = crop_and_rgb2gray(image).astype(np.float32)
  greyed *= (1.0 / 255.0)
  return greyed


def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def train():
  env = gym.make("Pong-v0")
  observation = prepro_grey(env.reset())
  rewards = []
  batch_rewards = []
  running_reward = None
  reward_sum = 0
  episode_number = 0
  frames = []
  actions = []
  current_framestack = np.zeros((80, 80, 4))
  # the new frame is placed in the front
  current_framestack[:,:,0] = observation
  
  input_ = tf.placeholder(dtype=tf.float32, shape=[None, 80, 80, 4])
  targets = tf.placeholder(tf.float32, (None, 1), name="targets")
  action_probs, logits = deep_q_net(input_)
  discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")

  # not really sure about this part - why are logits with the discounts?
  loss = tf.reduce_mean((discounted_rewards - logits) *  tf.nn.sigmoid_cross_entropy_with_logits(logits, targets))
  train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
  #gradients = optimizer.compute_gradients(loss)
  #train_op = optimizer.apply_gradients(gradients)

  sess = tf.Session()
  init = tf.initialize_all_variables()
  sess.run(init)
  
  while episode_number < 2000:
    env.render()
    # forward the policy network and sample an action from the returned probability
    action_prob = sess.run(action_probs, {input_ : current_framestack.reshape(1, 80, 80, 4)})
    action = 2 if np.random.uniform() < action_prob else 3 # roll the dice!
    
    observation, reward, done, info = env.step(action)
    observation = prepro_grey(observation)
    # np.roll rolls the array so that the previous last one is now the first one.
    current_framestack = np.roll(current_framestack, 1, 2)
    #overwrite with the new frame
    current_framestack[:,:,0] = observation
    y = 1 if action == 2 else 0
    actions.append(y)
    rewards.append(reward)
    frames.append(current_framestack.copy())
    
    if done: # an episode finished
      episode_number += 1
      epr = np.vstack(rewards)
      # compute the discounted reward backwards through time
      discounted = discount_rewards(epr).ravel()
      actions = np.array(actions, dtype=np.float32).reshape(len(actions), 1)
      stacked_frames = np.stack(frames)

      sess.run([train_op], feed_dict = {input_ : stacked_frames,
                                 targets : actions,
                                 discounted_rewards : discounted
                                 })

      reward_sum += np.sum(rewards)
      print("average rewards: %f" % (np.sum(rewards)))
      observation = prepro_grey(env.reset())
      frames = []
      actions = []
      rewards = []
        
      observation = prepro_grey(env.reset())
      current_framestack = np.zeros((80, 80, 4))
      current_framestack[:,:,0] = observation
      
train()