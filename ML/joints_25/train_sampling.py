# from tensorflow.python.keras import Sequential
# from tensorflow.python.keras import regularizers
from tensorflow.python.keras import optimizers
# from tensorflow.python.keras.layers import LSTM, CuDNNLSTM
# from tensorflow.python.keras.layers import Dense
# from tensorflow.python.keras.layers import Dropout
# from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import ModelCheckpoint
# import keras.metrics
import numpy as np
import pickle

from data_helper import reduce_joint_dimension, create_model, reform_to_sequence


sequence_length = 20 # timestep

#### Prepare data
path_save = "F:/Master Project/Dataset/Extract_Data/25 joints"
f_x = open(path_save+"/train_x.pickle",'rb')
f_y = open(path_save+"/train_y.pickle",'rb')
origin_train_x = pickle.load(f_x)
origin_train_y = np.array(pickle.load(f_y))

f_x = open(path_save+"/test_x.pickle",'rb')
f_y = open(path_save+"/test_y.pickle",'rb')
origin_test_x = pickle.load(f_x)
origin_test_y = np.array(pickle.load(f_y))

origin_train_x = reduce_joint_dimension(origin_train_x, '6')
origin_test_x = reduce_joint_dimension(origin_test_x, '6')


#### Prepare model
number_feature = origin_train_x[0].shape[-1] # last index
load_model = False
model = create_model(sequence_length, number_feature)
start_epoch = 0

if load_model:
    weights_path = 'keep/weight-sampling-01-0.95 - 6 joints.hdf5'    
    model.load_weights(weights_path)

sgd = optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False)
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
filepath="weight-sampling-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]


#### Prepare Test Set
test_x, test_y  = reform_to_sequence(origin_test_x, origin_test_y, 10000, sequence_length)


#### Train
num_epoch = 100

for i_ep in range(start_epoch+1,num_epoch):
    
    print('epoch: ', i_ep)
    train_x, train_y = reform_to_sequence(origin_train_x, origin_train_y, 20000, sequence_length)
    model.fit(train_x, train_y, epochs=start_epoch+1,
             validation_data=(test_x,test_y), callbacks=callbacks_list, initial_epoch=start_epoch)
    print("-----------------------")

