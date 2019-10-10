
import numpy as np
x = np.array([1]*25)
y = np.array([2]*25)
z = np.array([3]*25) 
one_frame = np.array([0.0]*75)
one_frame[0::3] = x
one_frame[1::3] = y
one_frame[2::3] = z

print(one_frame)