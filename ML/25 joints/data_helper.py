import numpy as np

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

def reduce_joint_dimension(data,choose_type):

    ## choose joints
    if choose_type == '6':
        choose_joints = np.array([ 8, 6, 5, 12, 10, 9]) - 1
    elif choose_type == '14':
        choose_joints = np.array([ 8, 6, 5, 12, 10, 9, 15, 14, 13, 19, 18, 17, 4, 3 ] ) - 1
    else:## choose all
        choose_joints = np.array(range(25))

    select_column = []
    for i in range(len(choose_joints)): # 14 body join( except waist)
        select_column.append(0 + 3*choose_joints[i]) # select x
        select_column.append(1 + 3*choose_joints[i]) # select y
        select_column.append(2 + 3*choose_joints[i]) # select z 

    ### select
    new_data = []
    # data shape (file,frame,75)
    for file in data:
        new_file = file[:,select_column]
        new_data.append(new_file)
    return new_data

def sampling_x(x, sequence_length):
    
    frames = x
    
    random_sample_range = 8 # sampling value is not more than 3

    # Randomly choose sample interval and start frame
    sample_interval = np.random.randint(1, random_sample_range + 1)

    start_i = np.random.randint(0, len(frames) - sample_interval * sequence_length + 1)

    # Extract frames as tensors
    image_sequence = []
    end_i = sample_interval * sequence_length + start_i
    for i in range(start_i, end_i, sample_interval):
        # image_path = frames[i]
        if len(image_sequence) < sequence_length:
            image_sequence.append(frames[i])
        else:
            break
    image_sequence = np.array(image_sequence)   
    return image_sequence

# Use for sampling and reforming data for sending to ML model
def reform_to_sequence(data_x, data_y, random_time, sequence_length, is_2steam=False):
    
    # if is_training:
    #     random_time = 20000
    #     output_x = np.zeros((len(data_x)*random_time, sequence_length, data_x[0].shape[-1]) ) #(len,timestep, 28)
        
    # else:        
    #     random_time = 10000
    #     output_x = np.zeros((len(data_x)*random_time, sequence_length, data_x[0].shape[-1]) ) #(len*random_time,timestep, 28)
    feature_number = data_x[0].shape[-1]
    output_x = np.zeros((len(data_x)*random_time, sequence_length, feature_number) ) #(len*random_time,timestep, num_feature)
    output_xdiff = np.zeros((len(data_x)*random_time, sequence_length, feature_number) )

    count = 0
    output_y = np.arange( len(data_y)*random_time ) # create array
    new_row = np.ones( (1, feature_number))
    # sampling window-data in random_time time
    for n_time in range(random_time):
        for i,x in enumerate(data_x):
            sequence = sampling_x(x, sequence_length)
            output_x[count] = sequence
            output_y[count] = data_y[i]
            if is_2steam:                
                diff_sequence = np.diff(sequence, n=1, axis=0)
                output_xdiff[count] = np.append(new_row, diff_sequence, axis=0)

            count += 1
    
    # output_x - x_data   : shape(num_of_file * random_time, sequence_length, 75)
    # output_y - y_data   : shape(num_of_file * random_time, sequence_length, 75)
    if is_2steam:
        return output_x, output_y, output_xdiff
    else:
        return output_x, output_y   


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


def create_2steam_model(num_frame, num_joint):

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

#     model = Sequential()
#     model.add(CuDNNLSTM(50, input_shape=(num_frame, num_joint),return_sequences=False))
#     model.add(Dropout(0.4))#使用Dropout函数可以使模型有更多的机会学习到多种独立的表征
#     model.add(Dense(60) )
#     model.add(Dropout(0.4))
#     model.add(Dense(4, activation='softmax'))

#     feature = concatenate([up_feature, down_feature])

#     fc_1 = Dense(units=256, activation='relu', use_bias=True, kernel_regularizer=l2(0.001))(feature)
#     fc_1 = Dropout(0.5)(fc_1)

#     fc_2 = Dense(units=128, activation='relu', use_bias=True)(fc_1)

#     fc_3 = Dense(units=96, activation='relu', use_bias=True)(fc_2)

#     fc_4 = Dense(units=4, activation='softmax', use_bias=True)(fc_3)

#     network = Model(input=[up_0, up_1, down_0, down_1], outputs=fc_4)
#     return network

#     return model

# # dd