"""
YAMC, yet another mnist classifier, is a project to learn the low-level tensorflow API
The data is taken from the website: http://yann.lecun.com/exdb/mnist/
"""

import numpy as np
import tensorflow as tf

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y

writer = tf.summary.FileWriter('../board/')
writer.add_graph(tf.get_default_graph())

sess = tf.Session()
print(sess.run(z, feed_dict={x:3, y:4}))
