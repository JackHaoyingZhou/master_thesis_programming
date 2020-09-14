import numpy as np
import scipy.interpolate
from scipy.linalg import expm
from pydmps.cs import CanonicalSystem


class DMPori(object):
	def __init__(self, n_dmps, n_bfs, dt=.01,y0=0, goal=1, w=None,ay=None, by=None, **kwargs):
		self.n_bfs = n_bfs
		self.n_dmps = n_dmps
		self.dt = dt
		#self.tau = tau

		self.n_dmps = self.n_dmps*3

		if isinstance(y0, (int, float)):
			y0 = np.ones(self.n_dmps)*y0
		self.y0 = y0
		if isinstance(goal, (int,float)):
			goal = np.ones(self.n_dmps)*goal
		self.goal = goal

		if w is None:
			# default is f = 0
			w = np.zeros(self.n_dmps, self.n_bfs)
		self.w = w
		self.ay = np.ones(n_dmps)*25.0 if ay is None else ay
		self.by = self.ay/4. if by is None else by

		self.cs = CanonicalSystem(dt=self.dt, **kwargs)
		self.cs.ax = 1.0
		self.timesteps = int(self.cs.run_time/self.dt)
		# print(self.timesteps)

		self.n_dmps = int(self.n_dmps/3)

		self.gen_centers()

		self.h = np.ones(self.n_bfs)*self.n_bfs**1.5/ self.c /self.cs.ax

		self.check_offset()

	def gen_centers(self):
		des_c = np.linspace(0, self.cs.run_time, self.n_bfs)

		self.c = np.ones(len(des_c))
		for i in range(len(des_c)):
			# finding x for desired time t
			self.c[i] = np.exp(-self.cs.ax*des_c[i])

	def check_offset(self):
		for d in range(self.n_dmps*3):
			#print(self.n_dmps)
			if self.y0[d] == self.goal[d]:
				self.goal[d] += 1e-4

	def gen_rmtx_x(self,theta):
		rmatrix_x = np.zeros((3,3))
		rmatrix_x[0,0] = 1
		rmatrix_x[1,1] = np.cos(theta)
		rmatrix_x[1,2] = -np.sin(theta)
		rmatrix_x[2,1] = np.sin(theta)
		rmatrix_x[2,2] = np.cos(theta)
		return rmatrix_x

	def gen_rmtx_y(self,theta):
		rmatrix_y = np.zeros((3,3))
		rmatrix_y[1,1] = 1
		rmatrix_y[0,0] = np.cos(theta)
		rmatrix_y[2,0] = -np.sin(theta)
		rmatrix_y[0,2] = np.sin(theta)
		rmatrix_y[2,2] = np.cos(theta)
		return rmatrix_y

	def gen_rmtx_z(self,theta):
		rmatrix_z = np.zeros((3,3))
		rmatrix_z[2,2] = 1
		rmatrix_z[0,0] = np.cos(theta)
		rmatrix_z[0,1] = -np.sin(theta)
		rmatrix_z[1,0] = np.sin(theta)
		rmatrix_z[1,1] = np.cos(theta)
		return rmatrix_z

	def gen_init(self, R_std):
		R_0 = [[] for _ in range(self.n_dmps)]
		for i in range(len(R_std)):
			R_0[i] = R_std[i][0]
		return R_0			

	def gen_goal(self, R_std):
		R_g = [[] for _ in range(self.n_dmps)]
		for i in range(len(R_std)):
			R_g[i] = R_std[i][-1]
		return R_g

	def logRmtx(self, R_g, R):
		R_t = np.dot(R_g, R.T)
		if np.all(R_t == np.eye(3)):
			omega = np.array([[0,0,0]]).T
		else:
			theta = np.arccos((np.trace(R_t)-1)/2)
			#print(theta)
			#print(R_t)
			#theta = np.nan_to_num(theta)
			# print(theta)
			n_o = np.array([(R_t[2,1]-R_t[1,2]),(R_t[0,2]-R_t[2,0]),(R_t[1,0]-R_t[0,1])])
			#print(n_o[0])
			if np.sin(theta)!=0:
				n = 1/(2*np.sin(theta))*n_o
				omega = theta*n
			else:
				omega = np.array([0,0,0])
			###### ^ maybe not right need to modify
			#print(n[0])
			#omega = theta*n
			#print(omega[0])
		return omega

	def omegalogR(self, R_g, R_o):
		omega_o = [[] for _ in range(self.timesteps)]
		for i in range(len(omega_o)):
			omega_o[i] = self.logRmtx(R_g, R_o[i])
		return omega_o

	def rmtxomega(self, omega_o):
		omega = np.ones([3,self.timesteps])
		for i in range(len(omega_o)):
			omega[:,i] = omega_o[i]
		return omega

	def calcfd(self, Romega, omega, domega,i):
		for i in range(1):
			fd = domega - self.ay[i]*(self.by[i]*Romega - omega)
			print(self.ay[i])
		return fd


	def gen_psi(self,x):
		if isinstance(x, np.ndarray):
			x = x[:,None]
		return np.exp(-self.h *(x-self.c)**2)

	def gen_weights(self, omega_o, f_d):
		x_track = self.cs.rollout()
		psi_track = self.gen_psi(x_track)
		w = [[] for _ in range(self.n_bfs)]
		s = np.ones((len(x_track),len(omega_o)))
		for j in range(len(x_track)):
			s[j,:] = x_track[j]*omega_o
		s = s.T
		for i in range(self.n_bfs):
			commtx = np.dot(s,np.diag(psi_track[:,i]))
			num = np.dot(commtx, f_d.T)
			dem = np.dot(commtx, s.T)
			#print(num)
			#print(dem)
			w[i] =np.dot(num, np.linalg.pinv(dem))
			w[i] = np.nan_to_num(w[i]) 
		return w

	def gen_f0(self, w, f_term):
		x_track = self.cs.rollout()
		psi_track = self.gen_psi(x_track)
		f0 = np.ones((3, self.timesteps))
		#f_term = f_term[:,np.newaxis]
		for i in range(self.timesteps):
			num = np.zeros((3,3))
			dem = 0
			for j in range(self.n_bfs):
				num = num + psi_track[i,j]*w[j]
				dem = dem + psi_track[i,j]
			f0[:,i] = (1/dem)*np.dot(num,f_term)*x_track[i]
		return f0

	def gen_psimtx(self, y_des):
		R_t = [[] for _ in range(self.n_dmps*3)]
		for i in range(len(R_t)):
			R_t[i] =[[] for _ in range(self.timesteps)]
			for j in range(len(R_t[i])):
				if (i+1)%3 == 1:
					R_t[i][j] = self.gen_rmtx_x(y_des[i,j])
				elif (i+1)%3 == 2:
					R_t[i][j] = self.gen_rmtx_y(y_des[i,j])
				else:
					R_t[i][j] = self.gen_rmtx_z(y_des[i,j])
		return R_t

	def gen_combmtx(self, R_t):
		R_o = [[] for _ in range(self.n_dmps)]
		for i in range(len(R_o)):
			R_o[i] = [[] for _ in range(self.timesteps)]
			for j in range(len(R_o[i])):
				R_oa = np.dot(R_t[i][j],R_t[i+1][j])
				R_o[i][j] = np.dot(R_oa, R_t[i+2][j])
		return R_o

	def imitate_ori(self, y_des):
		if y_des.ndim == 1:
			y_des = y_des.reshape(1, len(y_des))
		#self.y0 = y_des[:,0].copy()
		self.y_des = y_des.copy()
		#self.goal = self.gen_goal(y_des)
		#self.check_offset()

		#import scipy.interpolate
		#from scipy.linalg import expm
		path = np.zeros((self.n_dmps*3, self.timesteps))
		x = np.linspace(0, self.cs.run_time, y_des.shape[1])
		for d in range(self.n_dmps*3):
			path_gen = scipy.interpolate.interp1d(x, y_des[d])
			for t in range(self.timesteps):
				path[d,t] = path_gen(t*self.dt)
		theta_des = path

		# calculate derivatives
		dtheta_des = np.diff(theta_des)/self.dt
		# add zeros
		dtheta_des = np.hstack((np.zeros((self.n_dmps*3,1)),dtheta_des))
		## double d
		ddtheta_des = np.diff(dtheta_des)/self.dt
		# add zeros
		ddtheta_des = np.hstack((np.zeros((self.n_dmps*3,1)),ddtheta_des))

		## test
		#print(dtheta_des.shape)

		# generate rotation matrix for x,y,z
		R_t = self.gen_psimtx(y_des)
		# generate total rotation matrix for each dmp
		R_o = self.gen_combmtx(R_t)
		# save the standard rotation matrix to R_std
		R_std = R_o
		# generate goal and initial one
		R_0 = self.gen_init(R_std)
		R_g = self.gen_goal(R_std)
		# next step

		R1 = [[] for _ in range(self.n_dmps)]
		omega_o = [[] for _ in range(self.n_dmps)]
		omega_m = [[] for _ in range(self.n_dmps)]
		fd = [[] for _ in range(self.n_dmps)]
		f_o = [[] for _ in range(self.n_dmps)]
		self.w = [[] for _ in range(self.n_dmps)]
		R_out = [[] for _ in range(self.n_dmps)]
		yeta = [[] for _ in range(self.n_dmps)]
		yeta_dot = [[] for _ in range(self.n_dmps)]

		for i in range(self.n_dmps):
			R_out[i] = [[] for _ in range(self.timesteps)]
			yeta[i] = [[] for _ in range(self.timesteps)]
			yeta_dot[i] = [[] for _ in range(self.timesteps)]

			R1[i] = self.logRmtx(R_g[i], R_0[i])
			omega_o[i] = self.omegalogR(R_g[i], R_o[i])
			## convert list to matrices
			omega_m[i] = self.rmtxomega(omega_o[i])
			fd[i] = self.calcfd(omega_m[i], dtheta_des[(3*i):(3*(i+1)),:], ddtheta_des[(3*i):(3*(i+1)),:],i)
			self.w[i] = self.gen_weights(R1[i], fd[i])
			f_o[i] = self.gen_f0(self.w[i],R1[i])
			R_out[i][0] = R_std[i][0]
			yeta[i][0] = self.y_des[(3*i):(3*(i+1)),0]
		R2 = [[] for _ in range(self.timesteps)]

		for j in range(self.n_dmps):
			#print(R_g[j])
			for k in range(self.timesteps-1):
				#print(R_out[j][k])
				yeta_dot[j][k] = self.ay[j]*(self.by[j]*(self.logRmtx(R_g[j],R_out[j][k]))-yeta[j][k])+f_o[j][:,k]
				print(k)

				R2[k] = self.logRmtx(R_g[j],R_out[j][k])
				#print(yeta_dot[j][k])
				yeta[j][k+1] = yeta[j][k] + self.dt*yeta_dot[j][k]
				print(yeta[j][k+1])
				yeta_x = yeta[j][k][0]
				yeta_y = yeta[j][k][1]
				yeta_z = yeta[j][k][2]
				yeta_mtx = np.array([(0,-yeta_z,yeta_y),(yeta_z,0,-yeta_x),(-yeta_y,yeta_x,0)])
				print(yeta_mtx)
				R_out[j][k+1] = np.dot(expm(self.dt*yeta_mtx),R_out[j][k])
		R2[-1] = self.logRmtx(R_g[0],R_g[0])

		#x_track = self.cs.rollout()
		# print(x_track.shape)

		#psi_track = self.gen_psi(x_track)
		#print(np.dot(R_g[0],R_0[0]))
		return R_out,R_std,yeta, R2

# ============================
# Test Code
# ===========================

if __name__ == "__main__":

	import matplotlib.pyplot as plt

	y_des = np.load('gen_path16.npz')['A']
	### initialize
	y_des -= y_des[:,0][:,None]

	#print(y_des.shape)
	#print(y_des[0,:])

	dmpori = DMPori(n_dmps=1,n_bfs=100,dt=.005,ay = np.ones(1)*25,by= np.ones(1)*4,w=np.zeros((1,10)))

	R_out, R_std, yeta, R2 = dmpori.imitate_ori(y_des=y_des)

	tstep = int(1.0/0.005)
	y1 = np.zeros((9*1, tstep))
	y2 = np.zeros((9*1, tstep))
	y3 = np.zeros((3, tstep))
	y4 = np.zeros((3, tstep))

	time = np.linspace(0,1.0,tstep)

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

		y3[0,k] = yeta[0][k][0]
		y3[1,k] = yeta[0][k][1]
		y3[2,k] = yeta[0][k][2]

		y4[0,k] = [k][0]
		y4[1,k] = R2[k][1]
		y4[2,k] = R2[k][2]



	#print(R_out[0][199][0,0])
	print(R_std[0][199])
	print(R2[50][0])


	plt.figure(9, figsize=(8,6))

	plt.subplot(331)
	plt.plot(time,y1[0,:],time,y2[0,:])
	plt.legend(['dmp','original'])
	plt.xlabel('time(s)')
	plt.ylabel('R value')
	plt.grid()
	plt.subplot(332)
	plt.plot(time,y1[1,:],time,y2[1,:])
	plt.legend(['dmp','original'])
	plt.xlabel('time(s)')
	plt.ylabel('R value')
	plt.grid()
	plt.subplot(333)
	plt.plot(time,y1[2,:],time,y2[2,:])
	plt.legend(['dmp','original'])
	plt.xlabel('time(s)')
	plt.ylabel('R value')
	plt.grid()
	plt.subplot(334)
	plt.plot(time,y1[3,:],time,y2[3,:])
	plt.legend(['dmp','original'])
	plt.xlabel('time(s)')
	plt.ylabel('R value')
	plt.grid()
	plt.subplot(335)
	plt.plot(time,y1[4,:],time,y2[4,:])
	plt.legend(['dmp','original'])
	plt.xlabel('time(s)')
	plt.ylabel('R value')
	plt.grid()
	plt.subplot(336)
	plt.plot(time,y1[5,:],time,y2[5,:])
	plt.legend(['dmp','original'])
	plt.xlabel('time(s)')
	plt.ylabel('R value')
	plt.grid()
	plt.subplot(337)
	plt.plot(time,y1[6,:],time,y2[6,:])
	plt.legend(['dmp','original'])
	plt.xlabel('time(s)')
	plt.ylabel('R value')
	plt.grid()
	plt.subplot(338)
	plt.plot(time,y1[7,:],time,y2[7,:])
	plt.legend(['dmp','original'])
	plt.xlabel('time(s)')
	plt.ylabel('R value')
	plt.grid()
	plt.subplot(339)
	plt.plot(time,y1[8,:],time,y2[8,:])
	plt.legend(['dmp','original'])
	plt.xlabel('time(s)')
	plt.ylabel('R value')
	plt.grid()
	#print(yeta[0][0][0])

	# plt.figure(7, figsize=(8,6))
	# plt.subplot(311)
	# plt.plot(time,y3[0,:])
	# plt.subplot(312)
	# plt.plot(time,y3[1,:])
	# plt.subplot(313)
	# plt.plot(time,y3[2,:])

	# plt.figure(5, figsize=(8,6))
	# plt.subplot(311)
	# plt.plot(time,y4[0,:])
	# plt.subplot(312)
	# plt.plot(time,y4[1,:])
	# plt.subplot(313)
	# plt.plot(time,y4[2,:])
	plt.show()



	#print(np.diag(Rt.T))
