import numpy as np

def reduce_joint_dimension(data,choose_type):

    ## choose joints
    if choose_type == '6':
        choose_joints = np.array([ 8, 6, 5, 12, 10, 9]) - 1
    elif choose_type == '12': # 2 arm and hand
        choose_joints = np.array([ 22, 23, 7, 8, 6, 5, ## left
                                   24, 25, 11, 12, 10, 9, ## right
                                 ]) - 1
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
    
    ### frames is each file, each file has several frame 
    frames = x
    
    random_sample_range = 10 # sampling value is not more than 3

    ## change position 
    x_delta = 0.3
    x_random = np.random.uniform(-x_delta, x_delta, 1)
    z_delta = 0.7
    z_random = np.random.uniform(-z_delta, z_delta, 1)

    ## change height
    y_delta = 0.25
    y_random = np.random.uniform(1-y_delta, 1+y_delta, 1)

    # Randomly choose sample interval and start frame
    sample_interval = np.random.randint(1, random_sample_range + 1)

    start_i = np.random.randint(0, len(frames) - sample_interval * sequence_length + 1)

    # Extract frames as tensors
    image_sequence = []
    end_i = sample_interval * sequence_length + start_i
    for i in range(start_i, end_i, sample_interval):
        # image_path = frames[i]
        if len(image_sequence) < sequence_length:
            #frames[i] is one frame
            frames[i,0::3] = frames[i,0::3] + x_random # add position noise to x
            frames[i,2::3] = frames[i,2::3] + z_random # add position noise to z
            frames[i,1::3] = frames[i,1::3] * y_random # add height noise to y
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


# # dd