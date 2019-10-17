from keras import Sequential
from keras.layers import CuDNNLSTM, LSTM
# from tensorflow.python.keras.layers import CuDNNLSTM
from keras.layers import Dense, Input
from keras.layers import Dropout, concatenate, Flatten
from keras.regularizers import l2
from keras.models import Model


# from tensorflow.python.keras import Sequential
# from tensorflow.python.keras import regularizers
# from tensorflow.python.keras import optimizers
# from tensorflow.python.keras.layers import LSTM, CuDNNLSTM
# from tensorflow.python.keras.layers import Dense, Input
# from tensorflow.python.keras.layers import Dropout, concatenate
# from tensorflow.python.keras.regularizers import l2
# from tensorflow.python.keras.models import Model



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


def create_2stream_model(num_frame, num_joint):

    up = Input(shape=(num_frame, num_joint), name='up_stream')
    down = Input(shape=(num_frame, num_joint), name='down_stream')

    up_feature = CuDNNLSTM(64, return_sequences=False)(up)
    down_feature = CuDNNLSTM(64, return_sequences=False)(down)
    # up_feature = Flatten()(up)
    # down_feature = Flatten()(down)

    feature = concatenate([up_feature, down_feature])
    
    fc_1 = Dense(units=256, activation='relu', use_bias=True, kernel_regularizer=l2(0.001))(feature)
    fc_1 = Dropout(0.5)(fc_1)
    fc_2 = Dense(units=128, activation='relu', use_bias=True)(fc_1)
    fc_3 = Dense(units=96, activation='relu', use_bias=True)(fc_2)
    fc_4 = Dense(units=4, activation='softmax', use_bias=True, name='main_output')(fc_3)
    network = Model(inputs=[up,down], outputs=fc_4)
    return network
