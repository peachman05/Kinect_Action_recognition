# ---- read file -------
import re# 用于正则表达式的匹配操作，在问别人能解析、复杂字符串分析、信息提取是一个很有用的工具
import pandas as pd

# ---- plot graph ------
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import time


###################################################################
###---------------------  Read Data ----------------------------###
###################################################################

choose_joints = np.array([4, 21, 9, 10, 11, 5, 6, 7, 17, 18, 19, 13, 14, 15 ]) - 1 # start from 0
select_column = []
for i in range(14): # 14 body join( except waist)
    select_column.append(0 + 3*choose_joints[i]) # select x
    select_column.append(1 + 3*choose_joints[i]) # select y
    select_column.append(2 + 3*choose_joints[i]) # select z 

def reshape_data(file_name):
    data_total = []
    with open(file_name,'r',encoding='utf-8') as f:
        data = f.readlines()    
    ## extract data format
    data_temp = [] 
    index = 0
    for j in data:
        temp = re.findall(r"-*\d+\.?\d*",j)
        # print('1111111111111',temp)
        if temp != []:
            if index %25 == 0:
                data_temp.append([float(k) for k in temp])
            else:
                data_temp[-1].extend([float(k) for k in temp])
            index += 1
    print("tt")
    ## choose frame
    for m in range(len(data_temp)):
        data = np.array(data_temp[m])
        if(data.size) == 75:#3*25
            data_total.append(data)    
    print("bb")
    ## select column
    data_total=pd.DataFrame(data_total)
    data_total = data_total.loc[ : , select_column ] # [row, column]
    return data_total


file_name ='Kinet_JointPos_拍_10.txt'
data = reshape_data(file_name)
data = np.array(data)

###################################################################
###---------------------  Plot Graph ---------------------------###
###################################################################

# bone_list COCO 14 joint from 25
bone_list = [[7,6], [6,5], [5,1], [1,0], [1,2], [2,3], [3,4],
             [1,8], [1,11], [8,9], [9,10], [11,12], [12,13]]

def update_lines(num, data, lines, bone_list):    
    start_time = time.time()
    x = data[num, 0::3]
    y = data[num, 1::3]
    z = data[num, 2::3]
    for line, bone in zip(lines, bone_list):
        # NOTE: there is no .set_data() for 3 dim data...

        line.set_data([x[bone[0]], x[bone[1]]], [z[bone[0]], z[bone[1]]])
        line.set_3d_properties([y[bone[0]], y[bone[1]]])
    dif_t = (time.time() - start_time)
    if dif_t > 0:
        print("FPS: ", 1.0 / dif_t )
    return lines

fig = plt.figure()
ax = fig.gca(projection='3d')



x = data[0, 0::3]
y = data[0, 1::3]
z = data[0, 2::3]
lines = [ax.plot([x[bone[0]], x[bone[1]]], [z[bone[0]], z[bone[1]]], [y[bone[0]], y[bone[1]]])[0] for bone in bone_list]

ax.legend()    
ax.set_xlim3d(-0.5, 0.5)
ax.set_ylim3d(2.2, 2.7)
ax.set_zlim3d(0, 2)
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y') 

line_ani = animation.FuncAnimation(fig, update_lines, None, fargs=(data, lines, bone_list),
                                   interval=1, blit=False)

plt.show()
print("testeee")