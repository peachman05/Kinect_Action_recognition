import numpy as np

a = np.array([[1,2], [3,4]])
new = a.flatten()
b = np.array([5,6])
print(np.append(b,a))