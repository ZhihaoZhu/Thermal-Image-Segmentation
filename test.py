import numpy as np

x = np.ones((5,5,3))
x *= 1/np.max(x,axis=(0,1))
x = 1 - x @ np.array([1/3,1/3,1/3])
