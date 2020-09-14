from contcode import VrepEnvBase
import vrep
import numpy as np
from pydmps.cs import CanonicalSystem
import pydmps
import pydmps.dmp_discrete
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn

class MyClass(VrepEnvBase):

	def __init__(self, n_dmps, n_bfs, N_steps, ay=None, by=None, y0 = 0, goal = 1, w = None, ax = 1.0, y_des = None):
		super(MyClass, self).__init__()

		rc, self.sphere_handle = vrep.simxGetObjectHandle(self.clientID, 'Sphere', vrep.simx_opmode_blocking)

		vrep.simxGetObjectPosition(self.clientID, self.sphere_handle, -1,  vrep.simx_opmode_streaming)

		self.n_dmps = n_dmps
		self.n_bfs = n_bfs
		self.N_steps = N_steps
		self.dt = 1.0/self.N_steps
		self.y_des = y_des
		self.y0 = y0
		self.goal = goal
		if w is None:
			w = np.zeros((self.n_dmps, self.n_bfs))
		self.w = w
		self.ay = np.ones(n_dmps) * 25.0 if ay is None else ay # Schaal
		self.by = self.ay /4. if by is None else by
		## set CS systems
		self.cs = CanonicalSystem(dt = self.dt)
		self.cs.ax = ax

	def gen_centerf(self):
		des_c = np.linspace(0,1,self.n_bfs)
		self.c = np.ones(len(des_c))
		for n in range(len(des_c)):
			self.c[n] = np.exp(-self.cs.ax*des_c[n])
		self.h = np.ones(self.n_bfs)*self.n_bfs**1.5/self.c/self.cs.ax
		self.x = self.cs.rollout()
		return self.c, self.h, self.x

	def gen_psif(self, x):
		if isinstance(x, np.ndarray):
			x = x[:,None]
		return np.exp(-self.h*(x-self.c)**2)

	def run(self, y_pos):
		#rc, sphere_positions = 
		#print(y_pos)
		vrep.simxSetObjectPosition(self.clientID, self.sphere_handle, -1, y_pos,vrep.simx_opmode_blocking)
		self.synchronous_trigger()

		#print(sphere_positions)
		#print(sphere_positions[1])
		#return sphere_positions

if __name__ == "__main__":
	weights = np.load('weight1.npz')['A']
	#print(weights.shape)
	#print(weights.shape)
	#print(weights)
	y_des = np.load('path1.npz')['A']
	ay = 25.0
	by = 4.0

	y_des = np.transpose(y_des)
	y_des = y_des[:,68:]
	y_des = y_des[:,:-138]
	y_des[0,:] = np.linspace(0,1.0,len(y_des[0,:]))

	my_class = MyClass(n_dmps=3, n_bfs=100, N_steps=200, ay=25.0, by=4.0, y0 = y_des[:,0], goal=y_des[:,-1], w = weights,y_des = y_des)

	c,h,x = my_class.gen_centerf()

	N = 200
	dt = 1.0/N
	y0 = y_des[:,0]
	g = y_des[:,-1]
	y = np.zeros((3,N))
	y[0,0] = y0[0]
	y[1,0] = y0[1]
	y[2,0] = y0[2]
	my_class.run(y_pos=y[:,0])
	#print(y[0,1])
	dy = np.zeros((3,N))
	ddy = np.zeros((3,N))

	for i in range(1,N,1):
		#print(x[i])
		psi = my_class.gen_psif(x[i])
		den1 = np.dot(psi, weights[0])
		den2 = np.dot(psi, weights[1])
		den3 = np.dot(psi, weights[2])
		num = np.sum(psi)
		f1 = (den1/num)*x[i]*(g[0]-y0[0])
		f2 = (den2/num)*x[i]*(g[1]-y0[1])
		f3 = (den3/num)*x[i]*(g[2]-y0[2])
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
		my_class.run(y_pos=y[:,i])

