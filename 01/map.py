
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
f = x1 - 2 * x2


# In[3]:


sess = tf.Session()
print(sess.run(f, {x1: 8, x2: 3}))

