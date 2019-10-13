
import numpy as np

new_f = np.array([1]*75)
frame_window = np.empty((0,25*3))
# import pdb; pdb.set_trace()
frame_window = np.append(frame_window, np.reshape(new_f, (1,75)), axis=0 )

print(frame_window.shape)