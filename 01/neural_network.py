
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# In[2]:


def weight(shape = []):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)


# In[3]:


def bias(dtype = tf.float32, shape = []):
    initial = tf.zeros(shape, dtype = dtype)
    return tf.Variable(initial) 


# In[4]:


def sigmoid(x):
    return (1 /(1 + tf.exp(-x)))


# In[5]:


Q = 5
P = 2
R = 1


# In[6]:


sess = tf.InteractiveSession()


# In[7]:


X = tf.placeholder(dtype = tf.float32, shape = [None, Q])


# In[8]:


W1 = weight(shape = [Q, P])
b1 = bias(shape = [P])
f1 = tf.matmul(X, W1) + b1
sigm = sigmoid(f1)


# In[9]:


W2 = weight(shape = [P, R])
b2 = bias(shape = [R])
f2 = tf.matmul(sigm, W2) + b2


# In[10]:


init_op = tf.global_variables_initializer()
sess.run(init_op)


# In[11]:


y = sess.run(f2, {X: np.array([1,2,2,5,2]).astype(np.float32).reshape(1,5) })

print(y)

