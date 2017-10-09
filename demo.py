import numpy as np
import tensorflow as tf

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input1, input3)
mul = tf.multiply(input1, intermed)

with tf.Session() as sess:
  result = sess.run([intermed, mul])
  print(result)

# output:
# [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]