import numpy as np

a = np.load('weight_joy1_ori.npz')['A']
print(a.shape)
print(a[1][0])