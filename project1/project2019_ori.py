import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn
from dmpori import DMPori
from mtxtoangle import euler_angles_from_rotation_matrix as r_to_d
from transforms3d.euler import mat2euler
from transforms3d.euler import euler2mat

import pydmps
import pydmps.dmp_discrete

def modf_omega(omega_des):
	n_yrow = omega_des.shape[0]
	n_ycol = omega_des.shape[1]
	omega_mod = [[] for _ in range(n_yrow)]
	for i in range(len(omega_mod)):
		omega_mod[i] = np.zeros((3,n_ycol))
		if (i+1)%3 == 1:
			omega_mod[i][0,:] = omega_des[i,:]
		elif (i+1)%3 == 2:
			omega_mod[i][1,:] = omega_des[i,:]
		else:
			omega_mod[i][2,:] = omega_des[i,:]
	return omega_mod



#omega_des = np.load('./ori_gen_3.npz')['A']
# omega_des = np.load('./gen_path21.npz')['A']
#omega_des = np.load('./masterthesisdata/joy_gen1.npz')['B']
omega_des = np.loadtxt('ori_multi_4.txt')
# omega_des = np.loadtxt('ori_s_3.txt')
#omega_des = np.loadtxt('ori_1.txt')
print(omega_des.shape)
omega_des = np.transpose(omega_des)
#omega_des[:,0] = omega_des[:,0] + 0.1

#omega_des[2,:] = -omega_des[2,:]

#print(omega_des[:,500])
#omega_des -= omega_des[:, 0][:, None]

omega_mod = modf_omega(omega_des)


#print(omega_mod[0].shape)


#dmpori = DMPori(n_dmps=1,n_bfs=100,dt=.005,ay = np.ones(1)*5.0,by= np.ones(1)*4.0,w=np.zeros((1,10)))
# above a = 10, b = 4 original
### ay =25, by=4

n_yrow = omega_des.shape[0]
R_mtx = [[] for _ in range(n_yrow)]
R_mstd = [[] for _ in range(n_yrow)]
weight = [[] for _ in range(n_yrow)]
for i in range(len(omega_mod)):
	dmpori = DMPori(n_dmps=1,n_bfs=200,dt=1/200,ay = np.ones(1)*25.0,by= np.ones(1)*4.0,w=np.zeros((1,10)), ax = 1.0)
	R_mtx[i], R_mstd[i],weight[i] =  dmpori.imitate_ori(y_des = omega_mod[i])


np.savez_compressed('./weight_testr_multi_1',A=weight)
#np.savez_compressed('./weight_baxter_ori_3',A=weight)

#print(R_mtx[0][0][0].shape)
#tstep = int(1.0/0.005)

###### Need to be change !!!!!!!!!!!!
tstep = 200
tstep2 = omega_des.shape[1]

n_Ddmp = int(n_yrow/3)

R_out = [[] for _ in range(n_Ddmp)]
R_std = [[] for _ in range(n_Ddmp)]

for i in range(n_Ddmp):
	R_out[i] = [[] for _ in range(tstep)]
	R_std[i] = [[] for _ in range(tstep)]
	for j in range(tstep):
	#####RxRyRz
		R_out[i][j] = np.dot(np.dot(R_mtx[3*i][0][j],R_mtx[3*i+1][0][j]),R_mtx[3*i+2][0][j])
		R_std[i][j] = np.dot(np.dot(R_mstd[3*i][0][j],R_mstd[3*i+1][0][j]),R_mstd[3*i+2][0][j])

y1 = np.zeros((9*1, tstep))
y2 = np.zeros((9*1, tstep))
angle_1 = np.zeros((3, tstep))
angle_2 = np.zeros((3, tstep))

time = np.linspace(0,1.0,tstep)
time2 = np.linspace(0,1.0,tstep2)


for k in range(tstep):
	y1[0,k] = R_out[0][k][0,0]
	y1[1,k] = R_out[0][k][0,1]
	y1[2,k] = R_out[0][k][0,2]
	y1[3,k] = R_out[0][k][1,0]
	y1[4,k] = R_out[0][k][1,1]
	y1[5,k] = R_out[0][k][1,2]
	y1[6,k] = R_out[0][k][2,0]
	y1[7,k] = R_out[0][k][2,1]
	y1[8,k] = R_out[0][k][2,2]

	y2[0,k] = R_std[0][k][0,0]
	y2[1,k] = R_std[0][k][0,1]
	y2[2,k] = R_std[0][k][0,2]
	y2[3,k] = R_std[0][k][1,0]
	y2[4,k] = R_std[0][k][1,1]
	y2[5,k] = R_std[0][k][1,2]
	y2[6,k] = R_std[0][k][2,0]
	y2[7,k] = R_std[0][k][2,1]
	y2[8,k] = R_std[0][k][2,2]
	#angle_1[0,k],angle_1[1,k],angle_1[2,k] = r_to_d(R_std[0][k])
	#angle_2[0,k],angle_2[1,k],angle_2[2,k] = r_to_d(R_out[0][k])
	#angle_1[0,k],angle_2[1,k],angle_2[2,k] = mat2euler(R_out[0][k])
	#angle_1[0,k],angle_1[1,k],angle_1[2,k] = mat2euler(R_out[0][k])
	angle_1[0,k],angle_1[1,k],angle_1[2,k] = r_to_d(R_out[0][k])

# for ka in range(tstep2):
# 	R_s = euler2mat(-omega_des[0,ka],-omega_des[1,ka],-omega_des[2,ka])
# 	y2[0,ka] = R_s[0,0]
# 	y2[1,ka] = R_s[0,1]
# 	y2[2,ka] = R_s[0,2]
# 	y2[3,ka] = R_s[1,0]
# 	y2[4,ka] = R_s[1,1]
# 	y2[5,ka] = R_s[1,2]
# 	y2[6,ka] = R_s[2,0]
# 	y2[7,ka] = R_s[2,1]
# 	y2[8,ka] = R_s[2,2]

#np.savez_compressed('./ori_gen_s_3',A=angle_1)
print(angle_1.shape)
print(y2.shape)
print(y1.shape)


##### vrep comment#####
angle_1[2,:] = -angle_1[2,:]

plt.figure(2, figsize=(8,6))

plt.subplot(331)
plt.plot(time,y1[0,:],time,y2[0,:])
plt.legend(['dmp','original'])
plt.xlabel('time(s)')
plt.ylabel('R11 value')
plt.grid()
plt.subplot(332)
plt.plot(time,y1[1,:],time,y2[1,:])
plt.legend(['dmp','original'])
plt.xlabel('time(s)')
plt.ylabel('R12 value')
plt.grid()
plt.subplot(333)
plt.plot(time,y1[2,:],time,y2[2,:])
plt.legend(['dmp','original'])
plt.xlabel('time(s)')
plt.ylabel('R13 value')
plt.grid()
plt.subplot(334)
plt.plot(time,y1[3,:],time,y2[3,:])
plt.legend(['dmp','original'])
plt.xlabel('time(s)')
plt.ylabel('R21 value')
plt.grid()
plt.subplot(335)
plt.plot(time,y1[4,:],time,y2[4,:])
plt.legend(['dmp','original'])
plt.xlabel('time(s)')
plt.ylabel('R22 value')
plt.grid()
plt.subplot(336)
plt.plot(time,y1[5,:],time,y2[5,:])
plt.legend(['dmp','original'])
plt.xlabel('time(s)')
plt.ylabel('R23 value')
plt.grid()
plt.subplot(337)
plt.plot(time,y1[6,:],time,y2[6,:])
plt.legend(['dmp','original'])
plt.xlabel('time(s)')
plt.ylabel('R31 value')
plt.grid()
plt.subplot(338)
plt.plot(time,y1[7,:],time,y2[7,:])
plt.legend(['dmp','original'])
plt.xlabel('time(s)')
plt.ylabel('R32 value')
plt.grid()
plt.subplot(339)
plt.plot(time,y1[8,:],time,y2[8,:])
plt.legend(['dmp','original'])
plt.xlabel('time(s)')
plt.ylabel('R33 value')
plt.grid()

plt.show()
	
plt.figure(1)

plt.subplot(131)
plt.plot(time,angle_1[0,:],time2,-omega_des[0,:])
plt.legend(['dmp','original'])
plt.xlabel('time(s)')
plt.ylabel('alpha value')
plt.grid()
plt.subplot(132)
plt.plot(time,angle_1[1,:],time2,-omega_des[1,:])
plt.legend(['dmp','original'])
plt.xlabel('time(s)')
plt.ylabel('beta value')
plt.grid()
plt.subplot(133)
plt.plot(time,angle_1[2,:],time2,-omega_des[2,:])
plt.legend(['dmp','original'])
plt.xlabel('time(s)')
plt.ylabel('gama value')
plt.grid()

plt.show()