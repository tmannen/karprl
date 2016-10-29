import tensorflow as tf
import numpy as np
from scipy.misc import imresize

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

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * 0.99 + r[t]
    discounted_r[t] = running_add
  return discounted_r

def prepro_grey(I, new_size):
    """ crop and downsample, grayscale """
    #http://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    I = I[35:195, :]
    I = np.dot(I, [0.299, 0.587, 0.114])
    I = imresize(I, (new_size)).astype(np.float32) 
    I *= (1.0 / 255.0)

    return I