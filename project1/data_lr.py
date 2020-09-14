import numpy as np
from pydmps.cs import CanonicalSystem
import pydmps
import pydmps.dmp_discrete
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn

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


########## load file ########
# weights = np.load('weight_test_8.npz')['A']

weights = np.load('weight_test_multi_1.npz')['A']

#print(weights.shape)
print(weights.shape)
#print(weights)
#y_des = np.load('./masterthesisdata/joy_gen2.npz')['A']
#y_des = np.loadtxt('pos_4.txt')

# y_des_1 = np.loadtxt('pos_s_3.txt')

# y_des = np.loadtxt('pos_s_3.txt')

# y_des_1 = np.loadtxt('pos_4.txt')

# y_des = np.loadtxt('pos_4.txt')

y_des_1 = np.loadtxt('pos_multi_4.txt')

y_des = np.loadtxt('pos_multi_4.txt')

y_des = np.transpose(y_des)
y_des_1 = np.transpose(y_des_1)
# y_des = np.load('./gen_path21.npz')['A']

ay = 25.0
by = 4.0

##### for real data experienment######

#y_des = np.transpose(y_des)
#y_des = y_des[:,68:]
#y_des = y_des[:,:-138]
#y_des[0,:] = np.linspace(0,1.0,len(y_des[0,:]))
############


###### set parameters
N = 600
dt = 1.0/N
###### import canonical system
cs = CanonicalSystem(dt = dt,ax = 1.0)
cs.ax = cs.ax
x = cs.rollout()
#print(x.shape)
#print(x)
###### set goal 
#y0 = np.array([0.0011043,-0.07509588,0.1758978])
#g = np.array([1.00190017,1.68829869,0.08021972])
#y0 = y_des[:,0]
y0 = y_des[:,0]
g = y_des[:,-1]


# y0 = np.array([0.0,0.0,0.0])
# # g = np.array([0.0,0.0,0.0])+0.1
# g = np.array([2.0,2.0,2.0])
#g = np.array([-1.0,-1.0,-1.0])


# y0[0] = y0[0] + 0.02
# y0[1] = y0[1] + 0.03
# y0[2] = y0[2] - 0.02#+ 0.02

#g = y_des[:,-1]
# y_des[0,-1] = y_des[0,-1] - 0.1 #+ 0.05  #- 0.45 #+ 0.16 / 0.1 / 0.05
# y_des[1,-1] = y_des[1,-1] + 0.1 #+ 0.6 #- 0.2 #+ 0.6
# y_des[2,-1] = y_des[2,-1] ##### cannot change # -0.09 #+ 0.005



#######set numbef of basis functions
n_bfs = weights.shape[1]
###### calculate the centers
c = gen_c(n_bfs)
#print(c.shape)


###### calculate the h
h = np.ones(n_bfs)*n_bfs**1.5/c/cs.ax
#print(h.shape)

###### initalize
y = np.zeros((3,N))
y[0,0] = y0[0] - 0.01 #* 0.95##+0.05
y[1,0] = y0[1] + 0.02#* 0.95 ##+ 0.04
y[2,0] = y0[2] - 0.02#* 1.05#* - 0.02 #30.20
#print(y[0,1])
dy = np.zeros((3,N))
ddy = np.zeros((3,N))
#psi = np.exp(-h*(x[0]-c)**2)

# for i_dmpnum in range(3):
# 	for i_x in range(N):

for i in range(1,N,1):
	#print(x[i])
	psi = gen_psif(h,c,x[i])
	den1 = np.dot(psi, weights[0])
	den2 = np.dot(psi, weights[1])
	den3 = np.dot(psi, weights[2])
	num = np.sum(psi)
	f1 = (den1/num)*x[i]*(g[0]-y0[0])
	f2 = (den2/num)*x[i]*(g[1]-y0[1])
	f3 = (den3/num)*x[i]*(g[2]-y0[2])
	#print(psi)
	##### calculate the double derivative
		#####

	ddy[0,i] = ay*(by*(g[0]-y[0,i-1])-dy[0,i-1])+f1
	ddy[1,i] = ay*(by*(g[1]-y[1,i-1])-dy[1,i-1])+f2
	ddy[2,i] = ay*(by*(g[2]-y[2,i-1])-dy[2,i-1])+f3
	######
	dy[0,i] = ddy[0,i]*dt + dy[0,i-1]
	dy[1,i] = ddy[1,i]*dt + dy[1,i-1]
	dy[2,i] = ddy[2,i]*dt + dy[2,i-1]
	####### calculate the derivative
	###### calculate the distance
	y[0,i] = y[0,i-1] + dy[0,i]*dt 
	y[1,i] = y[1,i-1] + dy[1,i]*dt 
	y[2,i] = y[2,i-1] + dy[2,i]*dt 

#print(y[:,-1])
#print(num)

ax = plt.subplot(111, projection='3d')
#print(y_track)
#plt.plot(y_track[:,0], y_track[:, 1], c = 'r')
#ax.scatter(y_track[:,0],y_track[:,1],y_track[:,2],c='r')

ax.scatter(y[0,:], y[1,:],y[2,:], c = 'r')
ax.scatter(y_des_1[0,:], y_des_1[1,:],y_des_1[2,:], c = 'b')
plt.title('draw path')

#plt.axis('equal')
#plt.xlim([-2, 2])
#plt.ylim([-2, 2])
plt.legend(['moving target','original path' ])
ax.set_zlabel('z')
ax.set_ylabel('y')
ax.set_xlabel('x')
plt.show()

tstep1 = y_des.shape[1]#int(1.0/0.005)

tstep2 = N

#### 70
time1 = np.linspace(0,1.0,tstep1)
time2 = np.linspace(0,1.0,tstep2)

# plt.figure(3)
# plt.plot(y_des[1,:],y_des[2,:],y_track[:,1],y_track[:,2])
# plt.grid()
# plt.show()

plt.figure(2)
plt.subplot(131)
plt.plot(time2,y[0,:],time1,y_des_1[0,:])
plt.legend(['dmp','original'])
plt.xlabel('time(T)')
plt.ylabel('X')
plt.grid()
plt.subplot(132)
plt.plot(time2,y[1,:],time1,y_des_1[1,:])
plt.legend(['dmp','original'])
plt.xlabel('time(T)')
plt.ylabel('Y')
plt.grid()
plt.subplot(133)
plt.plot(time2,y[2,:],time1,y_des_1[2,:])
plt.legend(['dmp','original'])
plt.xlabel('time(T)')
plt.ylabel('Z')
plt.grid()
plt.show()

np.savez_compressed('./simple_m0_pos',A=y)

