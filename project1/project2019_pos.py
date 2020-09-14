import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn
from dmpori import DMPori
import random

import pydmps
import pydmps.dmp_discrete


####### Baxter input ############

y_des = np.loadtxt('pos_multi_4.txt')
# y_des = np.loadtxt('pos_s_3.txt')

y_des = np.transpose(y_des)


######### Function input ##########

# y_des = np.load('gen_path21.npz')['A']


########## mouse input? #########

#y_des = np.load('path2.npz')['A']

#y_des = np.transpose(y_des)


###### joy stick? ########

#y_des = np.load('./masterthesisdata/joy_gen4.npz')['A']

#y_des = np.load('./masterthesisdata/joypath1.npz')['A'] path = controller

#y_des = np.transpose(y_des)

############################################# remove the start/end
# y_des = y_des[:,10:]
# y_des = y_des[:,:-10]
# y_des[0,:] = np.linspace(0,1.0,len(y_des[0,:]))
#############################################

# for i in range(len(y_des[0,:])):
# 	y_des[0,i] = y_des[0,i] + random.uniform(-0.001,0.001)

#print(y_des[0,-1])

##
#print(y_des)

#y_des -= y_des[:, 0][:, None]  ##initialize

#print(y_des[:,-1])

print(y_des.shape)
#print(y_des[:,-1])
#print(y_des[:,0])

# test normal run
dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=3, n_bfs=500, ay=np.ones(3)*25.0,by=np.ones(3)*4.0,dt= 1/500, ax = 1.0)
#dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=1, n_bfs=500, ay=np.ones(1)*25.0,by=np.ones(1)*4.0,dt= 1/500)

y_des_m = y_des
# y_des_m[0,0] = y_des_m[0,0]+0.1
# y_des_m[1,0] = y_des_m[1,0]+0.1
# y_des_m[2,0] = y_des_m[2,0]+0.1

y_path, dy_des, ddy_des = dmp.imitate_path(y_des=y_des_m)

#print(dy_des.shape)

#plt.figure(1, figsize=(6,6))


y_track, dy_track, ddy_track, weight = dmp.rollout()

#print(y_track.shape)

np.savez_compressed('./weight_test_multi_1',A=weight)
#print(weight.shape)

ax = plt.subplot(111, projection='3d')
#print(y_track)
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

# dmpori = DMPori(n_dmps=1,n_bfs=100,dt=.005,ay = np.ones(1)*10.0,by= np.ones(1)*4.0,w=np.zeros((1,10)))
# # above a = 10, b = 4 original

# R_out, R_std = dmpori.imitate_ori(y_des=omega_des)
#print(y_track[0,:])
#print(y_track[-1,:])
#print(y_des[:,0])
#print(y_des[:,-1])
print(y_des.shape)

tstep1 = y_des.shape[1]#int(1.0/0.005)

tstep2 = 500 #$200

#### 70
time1 = np.linspace(0,1.0,tstep1)
time2 = np.linspace(0,1.0,tstep2)

plt.figure(3)
plt.plot(y_des[1,:],y_des[2,:],y_track[:,1],y_track[:,2])
plt.grid()
plt.show()



plt.figure(4)
plt.subplot(131)
plt.plot(time2,y_track[:,0],time1,y_des[0,:])
plt.legend(['dmp','original'])
plt.xlabel('time(s)')
plt.ylabel('X')
plt.grid()
plt.subplot(132)
plt.plot(time2,y_track[:,1],time1,y_des[1,:])
plt.legend(['dmp','original'])
plt.xlabel('time(s)')
plt.ylabel('Y')
plt.grid()
plt.subplot(133)
plt.plot(time2,y_track[:,2],time1,y_des[2,:])
plt.legend(['dmp','original'])
plt.xlabel('time(s)')
plt.ylabel('Z')
plt.grid()
# plt.subplot(334)
# plt.plot(time2,dy_track[:,0],time2,dy_des[0,:]/3.0)
# plt.legend(['dmp','original'])
# plt.xlabel('time(s)')
# plt.ylabel('vX')
# plt.grid()
# plt.subplot(335)
# plt.plot(time2,dy_track[:,1],time2,dy_des[1,:]/3.0)
# plt.legend(['dmp','original'])
# plt.xlabel('time(s)')
# plt.ylabel('vY')
# plt.grid()
# plt.subplot(336)
# plt.plot(time2,dy_track[:,2],time2,dy_des[2,:]/3.0)
# plt.legend(['dmp','original'])
# plt.xlabel('time(s)')
# plt.ylabel('vZ')
# plt.grid()
# plt.subplot(337)
# plt.plot(time2,ddy_track[:,0],time2,ddy_des[0,:]/9.0)
# plt.legend(['dmp','original'])
# plt.xlabel('time(s)')
# plt.ylabel('aX')
# plt.grid()
# plt.subplot(338)
# plt.plot(time2,ddy_track[:,1],time2,ddy_des[1,:]/9.0)
# plt.legend(['dmp','original'])
# plt.xlabel('time(s)')
# plt.ylabel('aY')
# plt.grid()
# plt.subplot(339)
# plt.plot(time2,ddy_track[:,2],time2,ddy_des[2,:]/9.0)
# plt.legend(['dmp','original'])
# plt.xlabel('time(s)')
# plt.ylabel('aZ')
# plt.grid()

plt.show()


# plt.figure(2, figsize=(8,6))