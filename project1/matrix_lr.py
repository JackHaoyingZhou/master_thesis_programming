import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn
import scipy.interpolate
from scipy.linalg import expm
from pydmps.cs import CanonicalSystem

from mtxtoangle import euler_angles_from_rotation_matrix as r_to_d

from transforms3d.euler import mat2euler
from transforms3d.euler import euler2mat

import pydmps
import pydmps.dmp_discrete

#####?
from dmpori import DMPori

def gen_rmtx_x(theta):
	rmatrix_x = np.zeros((3,3))
	rmatrix_x[0,0] = 1
	rmatrix_x[1,1] = np.cos(theta)
	rmatrix_x[1,2] = -np.sin(theta)
	rmatrix_x[2,1] = np.sin(theta)
	rmatrix_x[2,2] = np.cos(theta)
	return rmatrix_x

def gen_rmtx_y(theta):
	rmatrix_y = np.zeros((3,3))
	rmatrix_y[1,1] = 1
	rmatrix_y[0,0] = np.cos(theta)
	rmatrix_y[2,0] = -np.sin(theta)
	rmatrix_y[0,2] = np.sin(theta)
	rmatrix_y[2,2] = np.cos(theta)
	return rmatrix_y

def gen_rmtx_z(theta):
	rmatrix_z = np.zeros((3,3))
	rmatrix_z[2,2] = 1
	rmatrix_z[0,0] = np.cos(theta)
	rmatrix_z[0,1] = -np.sin(theta)
	rmatrix_z[1,0] = np.sin(theta)
	rmatrix_z[1,1] = np.cos(theta)
	return rmatrix_z

def d_to_r(theta):
	R_z = gen_rmtx_z(theta[2])
	R_y = gen_rmtx_y(theta[1])
	R_x = gen_rmtx_x(theta[0])
	R1 = np.dot(R_z,R_y)
	R = np.dot(R1,R_x)
	return R


def gen_c(n_bfs):
	des_c = np.linspace(0,1,n_bfs)
	c = np.ones(len(des_c))
	for n in range(len(des_c)):
		c[n] = np.exp(-cs.ax*des_c[n])
	return c

def gen_psif(h,c,x):
	if isinstance(x, np.ndarray):
		x = x[:,None]
	psi = np.exp(-h*(x-c)**2)
	return psi

def logRmtxn(R_g, R):
	R_t = np.dot(R_g, R.T)
	if np.all(R_t == np.eye(3)):
		omega = np.array([[0,0,0]]).T
	else:
		theta = np.arccos((np.trace(R_t)-1)/2)
		# print(theta)
		n_o = np.array([(R_t[2,1]-R_t[1,2]),(R_t[0,2]-R_t[2,0]),(R_t[1,0]-R_t[0,1])])
		#print(n_o[0])
		if np.sin(theta)!=0:
			n = 1/(2*np.sin(theta))*n_o
			omega = theta*n
		else:
			omega = np.array([0,0,0])
		###### ^ maybe not right need to modify
	return omega

def angles2mtx(theta):
	mtx_angle = euler2mat(theta[0],theta[1],theta[2])
	return mtx_angle

def mod_w(weights):
	n_bf = weights.shape[2]
	weight = [[] for _ in range(3)]
	for j in range(3):
		weight[j] = np.zeros((3,n_bf))
	for i in range(n_bf):
		weight[0][0,i] = weights[0][0][i][0,0]
		weight[1][1,i] = weights[1][0][i][1,1]
		weight[2][2,i] = weights[2][0][i][2,2]
	return weight

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

########## load file ########
#weights = np.load('weight_joy1_ori.npz')['A']
#weights = np.load('./weight_baxter_ori_3.npz')['A']
weights = np.load('weight_testr_multi_1.npz')['A']
print(weights.shape)
n_bfs = weights.shape[2]
weights = mod_w(weights)
#print(weights.shape)
print(weights[0].shape)
#print(weights[2][0][0])
#print(weights)
#y_des = np.load('./masterthesisdata/joy_gen1.npz')['B']
# y_des = np.transpose(y_des)

# y_des_1 = np.load('./gen_path21.npz')['A']

# y_des = np.load('./gen_path21.npz')['A']

y_des = np.loadtxt('ori_multi_4.txt')

y_des_1 = np.loadtxt('ori_multi_4.txt')

# y_des = np.loadtxt('ori_s_3.txt')

# y_des_1 = np.loadtxt('ori_s_3.txt')

# y_des = np.loadtxt('ori_4.txt')

# y_des_1 = np.loadtxt('ori_4.txt')

y_des = np.transpose(y_des)
y_des_1 = np.transpose(y_des_1)

print(y_des.shape)

# print(y_des_1)

####### change goal and initial state

y_des[0,0] = y_des[0,0] + (1/6) * np.pi#- (1/2) * np.pi #+ (1/4) * np.pi#+ 0.5#+ (1/10)*np.pi  
y_des[1,0] = y_des[1,0] + (1/6) * np.pi#- (1/3) * np.pi #+ 0.5#+ (1/10)*np.pi 
y_des[2,0] = y_des[2,0] + 0.03 #+ (1/2) * np.pi#+ (1/2) * np.pi#+ 0.03#+ (1/20)*np.pi 

# y_des[:,-1] = np.array([1.0,1.0,1.0])
# y_des[:,0] = np.array([0,0,0])

# y_des[:,-1] = np.array([3.0,3.0,3.0])
# y_des[:,0] = np.array([0,0,0])

# y_des[:,-1] = 2*y_des[:,-1]
# y_des[:,0] = y_des[:,0]
# y_des = y_des % (np.pi)


tstep = y_des.shape[1]
print(tstep)
y2 = np.zeros((9,tstep))

R_des = [[] for _ in range(tstep)]
for i_des in range(tstep):
	R_des[i_des] = d_to_r(y_des_1[:,i_des])

for i_p in range(tstep):
	y2[0,i_p] = R_des[i_p][0,0]
	y2[1,i_p] = R_des[i_p][0,1]
	y2[2,i_p] = R_des[i_p][0,2]
	y2[3,i_p] = R_des[i_p][1,0]
	y2[4,i_p] = R_des[i_p][1,1]
	y2[5,i_p] = R_des[i_p][1,2]
	y2[6,i_p] = R_des[i_p][2,0]
	y2[7,i_p] = R_des[i_p][2,1]
	y2[8,i_p] = R_des[i_p][2,2]

y_desm = modf_omega(y_des)

N = 600
dt = 1.0/N
###### import canonical system
cs = CanonicalSystem(dt = dt,ax=1.0)
x = cs.rollout()

ay = 25.0
by = 4.0
R_mtxm = [[] for _ in range(3)]
for j in range(3):
    g = y_desm[j][:,-1];#+0.1;
    #print(g)
    y_0 = y_desm[j][:,0];
    # y_0 = np.array([0,0,0])
    # g = np.array([1,1,1])
    R_g = d_to_r(g)
    R_0 = d_to_r(y_0)
    omega_02g = logRmtxn(R_g, R_0)
    #print(omega_02g)
    D_init = np.diag(omega_02g)
    #print(D_init)
    dy = np.zeros((3,N))
    y = np.zeros((3,N))
    R_mtxm[j] = [[] for _ in range(N)]
    omega = np.zeros((3,N))
    y[:,0] = y_0
    R_mtxm[j][0] = R_0
    #######set numbef of basis functions
    ###### calculate the centers
    c = gen_c(n_bfs)
    ###### calculate the h
    h = np.ones(n_bfs)*n_bfs**1.5/c/cs.ax
    for i_iter in range(1,N,1):
    	psi = gen_psif(h,c,x[i_iter])
    	#print(psi.shape)
    	dem = np.dot(weights[j],psi)
    	num = np.sum(psi)
    	#print(dem)
    	prev_mtx = dem/num
    	f_out = np.dot(prev_mtx, D_init)*x[i_iter]
    	#print(f_out)
    	#print(f_out.shape)
    	gamma_out = logRmtxn(R_g, R_mtxm[j][i_iter-1])
    	dy[:,i_iter] = ay*(by*gamma_out - y[:,i_iter-1]) + f_out
    	y[:,i_iter] = y[:,i_iter-1] + dy[:,i_iter-1]*dt
    	yeta_x = y[0,i_iter]
    	yeta_y = y[1,i_iter]
    	yeta_z = y[2,i_iter]
    	yeta_mtx = np.array([(0,-yeta_z,yeta_y),(yeta_z,0,-yeta_x),(-yeta_y,yeta_x,0)])
    	R_mtxm[j][i_iter] = np.dot(expm(dt*yeta_mtx),R_mtxm[j][i_iter-1])
#print(psi.shape)

R_mtx = [[] for _ in range(N)]
for i_mtx in range(N):
	R_a = np.dot(R_mtxm[2][i_mtx], R_mtxm[1][i_mtx])
	R_mtx[i_mtx] = np.dot(R_a, R_mtxm[0][i_mtx])

for i in range(len(R_mtx)):
	omega[0,i], omega[1,i], omega[2,i] = r_to_d(R_mtx[i])

time = np.linspace(0, 1.0, N)
time2 = np.linspace(0, 1.0, tstep)

y1 = np.zeros((9,N))
for i_plot in range(N):
	y1[0,i_plot] = R_mtx[i_plot][0,0]
	y1[1,i_plot] = R_mtx[i_plot][0,1]
	y1[2,i_plot] = R_mtx[i_plot][0,2]
	y1[3,i_plot] = R_mtx[i_plot][1,0]
	y1[4,i_plot] = R_mtx[i_plot][1,1]
	y1[5,i_plot] = R_mtx[i_plot][1,2]
	y1[6,i_plot] = R_mtx[i_plot][2,0]
	y1[7,i_plot] = R_mtx[i_plot][2,1]
	y1[8,i_plot] = R_mtx[i_plot][2,2]





plt.figure(1)

plt.subplot(131)
plt.plot(time,omega[0,:],time2,y_des_1[0,:])
plt.legend(['dmp','original'])
plt.xlabel('time(T)')
plt.ylabel('alpha value')
plt.grid()
plt.subplot(132)
plt.plot(time,omega[1,:],time2,y_des_1[1,:])
#plt.ylim(-3.5,3.5)
plt.legend(['dmp','original'])
plt.xlabel('time(T)')
plt.ylabel('beta value')
plt.grid()
plt.subplot(133)
plt.plot(time,omega[2,:],time2,y_des_1[2,:])
plt.legend(['dmp','original'])
plt.xlabel('time(T)')
plt.ylabel('gama value')
plt.grid()

plt.show()


plt.figure(2, figsize=(8,6))

plt.subplot(331)
plt.plot(time,y1[0,:],time2,y2[0,:])
plt.legend(['dmp','original'])
plt.xlabel('time(T)')
plt.ylabel('R11 value')
plt.grid()
plt.subplot(332)
plt.plot(time,y1[1,:],time2,y2[1,:])
plt.legend(['dmp','original'])
plt.xlabel('time(T)')
plt.ylabel('R12 value')
plt.grid()
plt.subplot(333)
plt.plot(time,y1[2,:],time2,y2[2,:])
plt.legend(['dmp','original'])
plt.xlabel('time(T)')
plt.ylabel('R13 value')
plt.grid()
plt.subplot(334)
plt.plot(time,y1[3,:],time2,y2[3,:])
plt.legend(['dmp','original'])
plt.xlabel('time(T)')
plt.ylabel('R21 value')
plt.grid()
plt.subplot(335)
plt.plot(time,y1[4,:],time2,y2[4,:])
plt.legend(['dmp','original'])
plt.xlabel('time(T)')
plt.ylabel('R22 value')
plt.grid()
plt.subplot(336)
plt.plot(time,y1[5,:],time2,y2[5,:])
plt.legend(['dmp','original'])
plt.xlabel('time(T)')
plt.ylabel('R23 value')
plt.grid()
plt.subplot(337)
plt.plot(time,y1[6,:],time2,y2[6,:])
plt.ylim(-1.1,1.1)
plt.legend(['dmp','original'])
plt.xlabel('time(T)')
plt.ylabel('R31 value')
plt.grid()
plt.subplot(338)
plt.plot(time,y1[7,:],time2,y2[7,:])
plt.ylim(-1.1,1.1)
plt.legend(['dmp','original'])
plt.xlabel('time(T)')
plt.ylabel('R32 value')
plt.grid()
plt.subplot(339)
plt.plot(time,y1[8,:],time2,y2[8,:])
plt.ylim(-1.1,1.1)
plt.legend(['dmp','original'])
plt.xlabel('time(T)')
plt.ylabel('R33 value')
plt.grid()

plt.show()

#np.savez_compressed('./simple_m2_ori',A=omega)