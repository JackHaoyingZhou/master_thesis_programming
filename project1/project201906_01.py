import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn

import pydmps
import pydmps.dmp_discrete

y_des = np.load('gen_path3.npz')['A']

#print(y_des[0,-1])

##
#print(y_des)

y_des -= y_des[:, 0][:, None]  ##initialize
#print(y_des)
##
print(y_des.shape)

# test normal run
dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=1, n_bfs=500, ay=np.ones(3)*10.0,by=np.ones(3)*4.0,dt= 0.005,imobj='orientation')

dmp.imitate_path(y_des=y_des)

#plt.figure(1, figsize=(6,6))
ax = plt.subplot(111, projection='3d')

y_track, dy_track, ddy_track = dmp.rollout(tau=1)
#plt.plot(y_track[:,0], y_track[:, 1], c = 'r')
ax.scatter(y_track[:,0],y_track[:,1],y_track[:,2],c='r')

ax.scatter(y_des[0,:], y_des[1,:],y_des[2,:], c = 'b')
plt.title('draw path')

#plt.axis('equal')
#plt.xlim([-2, 2])
#plt.ylim([-2, 2])
plt.legend(['moving target','original path' ])
ax.set_zlabel('z')
ax.set_ylabel('y')
ax.set_xlabel('x')
plt.show()