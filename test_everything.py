from keras import Sequential
from keras.layers import CuDNNLSTM, LSTM
# from tensorflow.python.keras.layers import CuDNNLSTM
from keras.layers import Dense, Input
from keras.layers import Dropout, concatenate, Flatten
from keras.regularizers import l2
from keras.models import Model

from keras import optimizers
from keras.callbacks import ModelCheckpoint
# import keras.metrics
import numpy as np

def create_model(num_frame, num_joint):
    model = Sequential()
    model.add(CuDNNLSTM(50, input_shape=(num_frame, num_joint),return_sequences=False))
    model.add(Dropout(0.4))#使用Dropout函数可以使模型有更多的机会学习到多种独立的表征
    model.add(Dense(256) )
    model.add(Dropout(0.4))
    model.add(Dense(64) )
    model.add(Dropout(0.4))
    model.add(Dense(4, activation='softmax'))
    return model

model = create_model(20, 18)
sgd = optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False)
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
filepath="weight-sampling-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]



# count = 0
def batch_generator(batch_size=16):
    '''
    Return a random image from X, y
    '''
    global count
    while True:
        # # choose batch_size random images / labels from the data
        idx = np.random.randint(0, batch_size, batch_size)
        # im = X[idx]
        # label = y[idx]

        # specgram = get_specgrams(im)


        # yield np.concatenate([specgram]), label
        yield np.zeros( (batch_size, 20, 18) ), idx


train_gen = batch_generator(batch_size=32)
valid_gen = batch_generator(batch_size=10000)


model.fit_generator(
    generator=train_gen,
    epochs=10,
    steps_per_epoch=20,
    validation_data=valid_gen,
    validation_steps=1,
    callbacks=callbacks_list,
    # use_multiprocessing=True,
    # workers=4,
)

# model.fit_generator(generate_arrays_from_file('/my_file.txt'),
#                     steps_per_epoch=10000, epochs=10)
