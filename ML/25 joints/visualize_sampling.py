# ---- plot graph ------
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D, proj3d
import matplotlib.animation as animation
import time

from data_helper import reform_to_sequence

import pickle

sequence_length = 20
num_joint = 25
bone_list = [[24,12], [25,12], [12,11], [11,10], [10,9], # right arm
            [22,8] ,[23,8], [8,7], [7,6], [6,5], # left arm
            [4,3], [3,21], [9,21], [5,21], [21,2], [2,1], [17,1], [13,1], # body
            [17,18], [18,19], [19,20], # right leg
            [13,14], [14,15], [15,16]]

bone_list = np.array(bone_list) - 1


###################################################################
###---------------------  Read Data  ---------------------------###
###################################################################
path_save = "F:/Master Project/Dataset/Extract_data/25 joints"
type_data = 'test'

f_x = open("{:}/{:}_x.pickle".format(path_save, type_data),'rb')
f_y = open("{:}/{:}_y.pickle".format(path_save,type_data),'rb')

test_x = pickle.load(f_x)
test_y = pickle.load(f_y)
test_x, test_y = reform_to_sequence(test_x, test_y, 1 , sequence_length)

###################################################################
###---------------------  Plot Graph ---------------------------###
###################################################################

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.legend()
ax.set_xlim3d(-0.2247708, 0.2919465)
ax.set_ylim3d(1.919823, 3)  # y in matplot is z in kinect
ax.set_zlim3d(-0.1041166, 1.431997)  # z in matplot is y in kinect
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')
annots = [ax.text2D(0,0,"POINT") for _ in range(num_joint)]
start_time = time.time()
def update_lines(num, data, lines, bone_list, my_ax): 
    global start_time

    dif_t = (time.time() - start_time)
    print(num)
    # print("Time taken : {0} seconds".format(dif_t))
    if dif_t > 0:
        print("FPS: ", 1.0 / dif_t )
    
    # kinect axis (z is deep)
    x = data[num, 0::3]
    y = data[num, 1::3]
    z = data[num, 2::3]
    # print("x min:", np.min(x)," x max", np.max(x)  )
    # print("z min:", np.min(z)," z max", np.max(z)  )
    # print("y min:", np.min(y)," y max", np.max(y)  )

    for line, bone in zip(lines, bone_list):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data([x[bone[0]], x[bone[1]]], [z[bone[0]], z[bone[1]]])  # x horizontal, z deep(y in matplot)
        line.set_3d_properties([y[bone[0]], y[bone[1]]]) # y vertical (z in matplot)

    for i, t in enumerate(annots):
        x_, y_, _ = proj3d.proj_transform(x[i], z[i], y[i], my_ax.get_proj())
        t.set_position((x_,y_))
        t.set_text(str(i+1))
   
    start_time = time.time()

    return lines, annots

x = np.array(range(num_joint))
y = np.array(range(num_joint))
z = np.array(range(num_joint))
lines = [ax.plot([x[bone[0]], x[bone[1]]],
                 [z[bone[0]], z[bone[1]]],
                 [y[bone[0]], y[bone[1]]])[0] for bone in bone_list]

print( len(test_x) )
num_video = 4
print(test_y)
line_ani = animation.FuncAnimation(fig, update_lines, test_x[num_video].shape[0],
                fargs=(test_x[num_video], lines, bone_list, ax),
                interval=1, blit=False)

# loop
plt.show()


print("test")