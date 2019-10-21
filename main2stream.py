# ---- read skelton ------- 
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

# ---- plot graph ------
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D, proj3d
import matplotlib.animation as animation
import time

from ML.joints_25.model_ML import create_2stream_model
import pickle


kinect_obj = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)
bodies = None


# choose_joints = np.array([4, 21, 9, 10, 11, 5, 6, 7, 17, 18, 19, 13, 14, 15 ]) - 1 # start from 0
# select_column = []
# for i in range(14): # 14 body join( except waist)
#     select_column.append(0 + 3*choose_joints[i]) # select x
#     select_column.append(1 + 3*choose_joints[i]) # select y
#     select_column.append(2 + 3*choose_joints[i]) # select z 

# # bone_list COCO 14 joint from 25
# bone_list = [[7,6], [6,5], [5,1], [1,0], [1,2], [2,3], [3,4],
#              [1,8], [1,11], [8,9], [9,10], [11,12], [12,13]]

# full bonelist 25 joints
bone_list = [[24,12], [25,12], [12,11], [11,10], [10,9], # right arm
            [22,8] ,[23,8], [8,7], [7,6], [6,5], # left arm
            [4,3], [3,21], [9,21], [5,21], [21,2], [2,1], [17,1], [13,1], # body
            [17,18], [18,19], [19,20], # right leg
            [13,14], [14,15], [15,16]]

bone_list = np.array(bone_list) - 1

# choose_joints = np.array([ 8, 6, 5, 12, 10, 9]) - 1
choose_joints = np.array([ 22, 23, 7, 8, 6, 5, ## left
                                   24, 25, 11, 12, 10, 9, ## right
                                 ]) - 1
# choose_joints = np.array([range(25)])

###################################################################
###-------------------  Action Recognition ---------------------###
###################################################################



weights_path = 'pretrain_model/up02_2steam-0.86-12j_15t.hdf5' # 15 frame
max_frame = 15
num_joint = 12
num_plot_joint = 25
num_feature = num_joint*3
# model = create_model(max_frame, num_joint*3, 4)
model = create_2stream_model(max_frame, num_feature)
model.load_weights(weights_path)

x_realtime_list = []

one_frame = np.array( [0.0]*num_feature )

threshold = 10
predict_queue = np.array([3] * threshold)
action_now = 3 # stand


frame_window = np.empty((0,num_feature))
new_row = np.ones( (1, num_feature))
def predict_action(new_f):
    # new_f = np.array([data_frame.ravel()])
    global frame_window, action_now
    new_f = np.reshape(new_f, (1,num_feature))
    frame_window = np.append(frame_window, new_f, axis=0 )
    if frame_window.shape[0] >= max_frame:

        diff_sequence = np.diff(frame_window, n=1, axis=0)
        frame_xdiff = np.append(new_row, diff_sequence, axis=0)
        
        frame_window_new = frame_window.reshape(1,max_frame, num_feature)
        frame_xdiff_new = frame_xdiff.reshape(1,max_frame, num_feature)


        result = model.predict([frame_window_new,frame_xdiff_new])
        frame_window = frame_window[1:max_frame]  
        # 拍球   投球   传球  其他动作
        # print("'dribble' 'throw' 'pass' 'stand")        
        class_label = ['dribble', 'shoot', 'pass', 'stand']
        v_ = result[0]
        predict_ind = np.argmax(v_)
        # print()
        predict_queue[:-1] = predict_queue[1:]
        predict_queue[-1] = predict_ind

        counts = np.bincount(predict_queue)
        # np.argmax(counts)
        if np.max(counts) >= threshold:
            action_now = np.argmax(counts)
        print( "{: <8}  {: <8}".format(class_label[predict_ind], class_label[action_now] ) )
        # print("拍球       投球       传球       其他动作")
        # print('{:.2f}    {:.2f}    {:.2f}    {:.2f}'.format(v_[0],v_[1],v_[2],v_[3]))


###################################################################
###---------------------  Plot Graph ---------------------------###
###################################################################

start_time = time.time()
count = 0
# call each timestep, num is number of that fram but we don't use it in here
def update_lines(num, kinect_obj, lines, bone_list, my_ax):    
    # global start_time
    # dif_t = (time.time() - start_time)
    # if dif_t > 0:
    #     print("FPS: ", 1.0 / dif_t )
    global count
    

    joints_data = read_skeleton(kinect_obj)
    if joints_data !=  None:
        x, y, z = joints_data

        # # one_frame_temp = np.array( [0.0]*num_plot_joint*3 )
        # # one_frame_temp[0::3] = x
        # # one_frame_temp[1::3] = y
        # # one_frame_temp[2::3] = z
        # # x_realtime_list.append(one_frame_temp)
        # # print(count)
        # count += 1

        # x_reduce = 
        # y_reduce = 
        # z_reduce = 
        
        one_frame[0::3] = x[choose_joints]
        one_frame[1::3] = y[choose_joints]
        one_frame[2::3] = z[choose_joints]
        predict_action(one_frame)
        

        #x = x *(-1) # mirror image
        # print("x min:", np.min(x)," x max", np.max(x)  )
        # print("y min:", np.min(y)," y max", np.max(y)  )
        # print("z min:", np.min(z)," z max", np.max(z)  )

        for line, bone in zip(lines, bone_list):
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_data([x[bone[0]], x[bone[1]]], [z[bone[0]], z[bone[1]]])
            line.set_3d_properties([y[bone[0]], y[bone[1]]])

        for i, t in enumerate(annots):
            x_, y_, _ = proj3d.proj_transform(x[i], z[i], y[i], my_ax.get_proj())
            t.set_position((x_,y_))
            t.set_text(str(i+1))

    
    # start_time = time.time()

    return lines, annots


def read_skeleton(kinect):
    if kinect.has_new_body_frame(): 
        bodies = kinect.get_last_body_frame()        
        if bodies is not None:             
            for i in range(0, kinect.max_body_count):
                body = bodies.bodies[i]
                if not body.is_tracked: 
                    continue                 
                joints = body.joints   
                x = np.array([0.0]*25)
                y = np.array([0.0]*25)
                z = np.array([0.0]*25)

                for j in range(25): # _Joint
                    # print( "type :", joints[j].JointType )
                    coor_point = joints[j].Position  
                    x[j] = coor_point.x
                    y[j] = coor_point.y + 1.1 # for adjust the tripod
                    z[j] = coor_point.z
                    
                
                # x = x[choose_joints]
                # y = y[choose_joints]
                # z = z[choose_joints]

                return (x, y, z)   
    return None


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.legend()
ax.set_xlim3d(-0.8, 0.8)
ax.set_ylim3d(0, 3) # z-kinect
ax.set_zlim3d(0, 1.4) # y-kinect
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')
annots = [ax.text2D(0,0,"POINT") for _ in range(num_plot_joint)]

x = np.array(range(num_plot_joint))
y = np.array(range(num_plot_joint))
z = np.array(range(num_plot_joint))
lines = [ax.plot([x[bone[0]], x[bone[1]]],
                 [z[bone[0]], z[bone[1]]],
                 [y[bone[0]], y[bone[1]]])[0] for bone in bone_list]



line_ani = animation.FuncAnimation(fig, update_lines, None,
                                   fargs=(kinect_obj, lines, bone_list, ax),
                                   interval=1, blit=False)

# loop
plt.show()

# end of program
kinect_obj.close()

### save file
f_x = open("skeleton_realtime.pickle",'wb')
pickle.dump(x_realtime_list,f_x)  # (num_of_file, all_frame, 75) - frame data    
f_x.close()
print("finish save file!!!")