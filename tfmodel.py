import tensorflow as tf

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

"""
class DeepQNet(object):
    def __init__(self, ):
        input_ = tf.placeholder(dtype=tf.float32, shape=[None, 80, 80, 4])
        self.model = model()
    def train(self):

    def model(self, input_):
        h0 = conv2d(input_, 16, 8, 8, 4, 4, name='h0_conv')
        h1 = conv2d(h0, 32, 4, 4, 2, 2, name='h1_conv')
        h2 = fc_relu(tf.reshape(h1, shape=[-1, 3200]), 256, name='h2_relu')
        h3 = fc_linear(h2, 1, name='h3_linear')
        return tf.nn.sigmoid(h3), h3
"""

def deep_q_net(input_):
    h0 = conv2d(input_, 16, 8, 8, 4, 4, name='h0_conv')
    h1 = conv2d(h0, 32, 4, 4, 2, 2, name='h1_conv')
    h2 = fc_relu(tf.reshape(h1, shape=[-1, 3200]), 256, name='h2_relu')
    h3 = fc_linear(h2, 1, name='h3_linear')
    return tf.nn.sigmoid(h3), h3

