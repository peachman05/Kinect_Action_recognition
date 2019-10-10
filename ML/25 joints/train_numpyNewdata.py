#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.layers import LSTM, CuDNNLSTM
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import numpy as np
import keras 
from numpy.lib.format import open_memmap


# In[2]:


# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# keras.backend.set_session(sess)


# In[3]:


path_data = 'F:/Master Project/Dataset/Extract_data/25 joints/'
train_X = np.load(path_data+'train_x.npy', mmap_mode='r')
train_Y = np.load(path_data+'train_y.npy', mmap_mode='r')
print(train_X.shape)


# In[4]:


# train_X[100][10]


# In[5]:


test_X = np.load(path_data+'test_x.npy', mmap_mode='r')
test_Y = np.load(path_data+'test_y.npy', mmap_mode='r')
print(test_X.shape)


# In[6]:


def create_model(num_frame, num_joint, num_output):
    model = Sequential()
    model.add(CuDNNLSTM(50, input_shape=(num_frame, num_joint),return_sequences=False))
    model.add(Dropout(0.4))#使用Dropout函数可以使模型有更多的机会学习到多种独立的表征
#     model.add(CuDNNLSTM(50, input_shape=(num_frame, num_joint)))
    model.add(Dense(64))
    model.add(Dropout(0.4))
    model.add(Dense(num_output, activation='softmax'))
    return model


# In[7]:


load_model = False
max_frame = test_X.shape[1]
number_column = test_X.shape[2]
model = create_model(max_frame, number_column, 4)
start_epoch = 0

if load_model:
    weights_path = 'weights-improvement-04-0.75.hdf5'
    
    model.load_weights(weights_path)

sgd = optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False)
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])    
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, )
callbacks_list = [checkpoint]


history = model.fit(train_X[:1000], train_Y[:1000], epochs=200,
          validation_data=(test_X,test_Y), callbacks=callbacks_list) #, initial_epoch=start_epoch)

print("tttt")
