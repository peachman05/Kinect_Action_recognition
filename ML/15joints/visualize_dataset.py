# ---- plot graph ------
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D, proj3d
import matplotlib.animation as animation
import time

import pickle

num_joint = 14
bone_list = [
                [1, 2], [2, 3], # left arm
                [4, 5], [5, 6], # right arm
                [9, 8], [8, 7], # left leg
                [12, 11], [11, 10],#right leg
                [14, 13], # neck
                [3, 14],  [14, 6], [6, 12], [12, 9], [9,3] # middle 
             ]

bone_list = np.array(bone_list) - 1


###################################################################
###---------------------  Read Data  ---------------------------###
###################################################################
path_save = "F:/Master Project/Dataset/Extract_data/15 joints"
type_data = 'test'

f_x = open("{:}/{:}_x.pickle".format(path_save, type_data),'rb')
f_y = open("{:}/{:}_y.pickle".format(path_save,type_data),'rb')

test_x = pickle.load(f_x)
test_y = pickle.load(f_y)


###################################################################
###---------------------  Plot Graph ---------------------------###
###################################################################

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.legend()
ax.set_xlim3d(-0.858405, 1.248923)
ax.set_ylim3d(-1.181592, 0.654658)  # y in matplot is z in kinect
ax.set_zlim3d(-0.30747,4.347053)  # z in matplot is y in kinect
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')

annots = [ax.text2D(0,0,"POINT") for _ in range(num_joint)]
start_time = time.time()
def update_lines(num, data, lines, bone_list, my_ax):  
    # kinect axis (z is deep)
    # ax.clear()

    global start_time
    dif_t = (time.time() - start_time)
    # if dif_t > 0:
    #     print("FPS: ", 1.0 / dif_t )

    x = data[num, 0::3] * -1
    y = data[num, 1::3]
    z = data[num, 2::3] * -1
    # print("x min:", np.min(x)," x max", np.max(x)  )
    # print("z min:", np.min(y)," z max", np.max(y)  )
    # print("y min:", np.min(z)," y max", np.max(z)  )

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
    # return lines, annotation





x = np.array(range(num_joint))
y = np.array(range(num_joint))
z = np.array(range(num_joint))
lines = [ax.plot([x[bone[0]], x[bone[1]]],
                 [z[bone[0]], z[bone[1]]],
                 [y[bone[0]], y[bone[1]]])[0] for bone in bone_list]


num_video = 1000
print(len(test_x))
print(test_y[num_video])

line_ani = animation.FuncAnimation(fig, update_lines, test_x[num_video].shape[0],
                fargs=(test_x[num_video], lines, bone_list, ax),
                interval=1, blit=False)

# loop
plt.show()


print("test")