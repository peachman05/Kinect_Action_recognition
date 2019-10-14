import numpy as np

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import CuDNNLSTM
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout


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
    
    random_sample_range = 3 # sampling value is not more than 3

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
def reform_to_sequence(data_x, data_y, is_training, sequence_length):
    
    if is_training:
        random_time = 20000
        output_x = np.zeros((len(data_x)*random_time, sequence_length, data_x[0].shape[-1]) ) #(len,timestep, 28)
        
    else:        
        random_time = 10000
        output_x = np.zeros((len(data_x)*random_time, sequence_length, data_x[0].shape[-1]) ) #(len*random_time,timestep, 28)
    
    count = 0
    output_y = np.arange( len(data_y)*random_time ) # create array
    
    # sampling window-data in random_time time
    for n_time in range(random_time):
        for i,x in enumerate(data_x):
            sequence = sampling_x(x, sequence_length)
            output_x[count] = sequence
            output_y[count] = data_y[i]
            count += 1
    
    # output_x - x_data   : shape(num_of_file * random_time, sequence_length, 75)
    # output_y - y_data   : shape(num_of_file * random_time, sequence_length, 75)
    return output_x, output_y   


def create_model(num_frame, num_joint):
    model = Sequential()
    model.add(CuDNNLSTM(50, input_shape=(num_frame, num_joint),return_sequences=False))
    model.add(Dropout(0.4))#使用Dropout函数可以使模型有更多的机会学习到多种独立的表征
    model.add(Dense(60) )
    model.add(Dropout(0.4))
    model.add(Dense(4, activation='softmax'))
    return model



# dd