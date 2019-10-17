

from keras.callbacks import ModelCheckpoint
from keras import optimizers
# import keras.metrics
import numpy as np
import pickle

from data_helper import reduce_joint_dimension, reform_to_sequence
from model_ML import create_2stream_model




sequence_length = 15 # timestep
try_detail = "12j_15t"

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

number_joint = 12

origin_train_x = reduce_joint_dimension(origin_train_x, str(number_joint))
origin_test_x = reduce_joint_dimension(origin_test_x, str(number_joint))


#### Prepare model
number_feature = origin_train_x[0].shape[-1] # last index
load_model = False
model = create_2stream_model(sequence_length, number_feature)
start_epoch = 1

if load_model:
    weights_path = 'keep/weight-2steam-01-0.95 - 6 joints.hdf5'    
    model.load_weights(weights_path)

sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
filepath="weight-2steam-{val_accuracy:.2f}-"+ try_detail +".hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=2, save_best_only=False)
callbacks_list = [checkpoint]

print(model.summary())

#### Prepare Test Set
test_x, test_y, test_xdiff  = reform_to_sequence(origin_test_x, origin_test_y, 10000, sequence_length, is_2steam=True)


### batch Generator

# def train_generator(origin_train_x, origin_train_y, batch_size=16):
#     while True:
#         train_x, train_y, train_xdiff = reform_to_sequence(origin_train_x, origin_train_y, batch_size, sequence_length, is_2steam=True)  
#         yield [train_x, train_xdiff], train_y

# def validate_generator(test_x, test_y, test_xdiff, batch_size=128):
#     while True:
#         yield  [test_x[:batch_size], test_xdiff[:batch_size]], test_y[:batch_size]



#### Train
num_epoch = 100
# step_per_epoch = 100

# batch_size = 32
# real_batch = batch_size * len(origin_test_y) ## 32 * 7
# train_gen = train_generator( origin_train_x, origin_train_y, batch_size=real_batch )
# valid_gen = validate_generator(test_x, test_y, test_xdiff, batch_size=1024)


# model.fit_generator(
#     generator=train_gen,
#     epochs=num_epoch,
#     steps_per_epoch=step_per_epoch,
#     validation_data=valid_gen,
#     validation_steps=1,
#     callbacks=callbacks_list,
# )




for i_ep in range(start_epoch+1,num_epoch):
    
    print('epoch: ', i_ep)
    train_x, train_y, train_xdiff = reform_to_sequence(origin_train_x, origin_train_y, 5000, sequence_length, is_2steam=True)
    # model.fit({'up_stream': train_x, 'down_stream': train_x},
    #           {'main_output': train_y})
    model.fit([train_x, train_xdiff], train_y, epochs=start_epoch+1,
             validation_data=([test_x,test_xdiff],test_y), callbacks=callbacks_list, initial_epoch=start_epoch)
    print("-----------------------")

