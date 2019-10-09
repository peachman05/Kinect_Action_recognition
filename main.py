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


kinect_obj = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)
bodies = None

num_joint = 25

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


###################################################################
###---------------------  Plot Graph ---------------------------###
###################################################################


# call each timestep, num is number of that fram but we don't use it in hear
def update_lines(num, kinect_obj, lines, bone_list, my_ax):    
    # start_time = time.time()
    joints_data = read_skeleton(kinect_obj)
    if joints_data !=  None:
        x, y, z = joints_data
        #x = x *(-1) # mirror image
        # print("x min:", np.min(x)," x max", np.max(x)  )
        # print("y min:", np.min(y)," y max", np.max(y)  )
        # print("z min:", np.min(z)," z max", np.max(z)  )

        for line, bone in zip(lines, bone_list):
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_data([x[bone[0]], x[bone[1]]], [z[bone[0]], z[bone[1]]])
            line.set_3d_properties([y[bone[0]], y[bone[1]]])

        # for i, t in enumerate(annots):
        #     x_, y_, _ = proj3d.proj_transform(x[i], z[i], y[i], my_ax.get_proj())
        #     t.set_position((x_,y_))
        #     t.set_text(str(i+1))

    # dif_t = (time.time() - start_time)
    # if dif_t > 0:
    #     print("FPS: ", 1.0 / dif_t )

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
                    y[j] = coor_point.y
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
ax.set_ylim3d(0.5, 1.5)
ax.set_zlim3d(-1.2, 0.2)
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')
annots = [ax.text2D(0,0,"POINT") for _ in range(num_joint)]

x = np.array(range(num_joint))
y = np.array(range(num_joint))
z = np.array(range(num_joint))
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
            