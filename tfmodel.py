import tensorflow as tf
import tensorflow.contrib.slim as slim
""" Some pointers from https://github.com/carpedm20/DCGAN-tensorflow """

def conv2d(input_, output_dim, filter_w, filter_h, stride_w, stride_h, name):
    with tf.variable_scope(name):
        #this kind of structure not really needed here since the model is not reused?
        w = tf.get_variable('w', [filter_h, filter_w, input_.get_shape()[-1], output_dim],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        conv = tf.nn.conv2d(input_, w, strides=[1, stride_h, stride_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        return tf.nn.relu(conv + biases)

def fc_relu(input_, output_dim, name):
    with tf.variable_scope(name):
        weights = tf.get_variable('w', [input_.get_shape()[-1], output_dim], 
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        return tf.nn.relu(tf.matmul(input_, weights) + biases)

def fc_linear(input_, output_dim, name):
    with tf.variable_scope(name):
        weights = tf.get_variable('w', [input_.get_shape()[-1], output_dim], 
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        return tf.matmul(input_, weights) + biases

"""
From https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf:
We now describe the exact architecture used for all seven Atari games. The input to the neural
network consists is an 84 × 84 × 4 image produced by φ. The first hidden layer convolves 16 8 × 8
filters with stride 4 with the input image and applies a rectifier nonlinearity [10, 18]. The second
hidden layer convolves 32 4 × 4 filters with stride 2, again followed by a rectifier nonlinearity. The
final hidden layer is fully-connected and consists of 256 rectifier units. The output layer is a fullyconnected
linear layer with a single output for each valid action. The number of valid actions varied
between 4 and 18 on the games we considered. We refer to convolutional networks trained with our
approach as Deep Q-Networks (DQN).
"""
def prepro_grey(image):
  """ crop and downsample, grayscale """
  #http://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
  def crop_and_rgb2gray(I):
    I = I[35:195] # crop
    if frame_size[0] == 40:
      I = I[::4,::4,:] # downsample by factor of 4
    else:
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

def deep_q_net(input_, num_actions):
    h0 = conv2d(input_, 16, 5, 5, 2, 2, name='h0_conv')
    h1 = conv2d(h0, 32, 5, 5, 2, 2, name='h1_conv')
    h2 = fc_relu(tf.reshape(h1, shape=[-1, 3200]), 256, name='h2_relu')
    # these of course have to be different because they have different weights..
    action_logits = fc_linear(h2, num_actions, name='action_logits')
    value_function = fc_linear(h2, 1, name='values')
    return tf.nn.softmax(action_logits), action_logits, value_function

class DeepQNet(object):
    def __init__(self, env, num_actions, input_shape, sess, learning_rate):
        self.env = env
        self.num_actions = num_actions
        self.input_shape = input_shape
        self.session = sess
        self.action_prob = None
        self.learning_rate = learning_rate
        self.train_op = None
        self.init_tf_variables()

    def init_tf_variables(self):
        width, height, depth = self.input_shape
        input_ = tf.placeholder(dtype=tf.float32, shape=[None, width, height, depth])
        actions = tf.placeholder(tf.float32, (None, self.num_actions), name="actions")
        discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")
        action_probs, action_logits, value_function = deep_q_net(input_)
        self.action_prob = action_probs #stored for sampling an action in an episode
        loss = tf.reduce_mean((discounted_rewards - values) * \
                tf.nn.sparse_softmax_cross_entropy_with_logits(action_logits, actions))
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(loss)
        self.session(tf.initialize_all_variables()).run()

    def run_episode(self):
        observations = [prepro_grey(self.env.reset())]
        rewards = []
        actions = []


    def train(self):