import numpy as np
from tfmodel import *

testdata = np.ones((1, 80, 80, 4))

input_ = tf.placeholder(dtype=tf.float32, shape=[None, 80, 80, 4])
deepq = deep_q_net(input_)
feed_dict = {input_: testdata}

sess = tf.Session()
tf.initialize_all_variables().run(session=sess)
prediction = sess.run(deepq, feed_dict)

print(prediction)