import tensorflow as tf

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
f = x1 - 2 *x2

sess = tf.Session()
print(sess.run(f, {x1: 8, x2: 3}))

