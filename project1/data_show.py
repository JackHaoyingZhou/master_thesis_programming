import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn
from dmpori import DMPori

import pydmps
import pydmps.dmp_discrete

y_des = np.load('joy_gen4.npz')['A']

y_des = np.transpose(y_des)
#print(y_des[0,-1])

##
print(y_des.shape)

print(y_des)

# y_des -= y_des[:, 0][:, None]  ##initialize