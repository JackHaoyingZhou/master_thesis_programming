import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


class ZHYDMPs():
	"""DMPs write by ZHY"""
	def __init__(self,filename,alpha,beta,ax,tau):
		self.filename = filename
		self.ax = ax
		self.alpha = alpha
		self.beta = beta
		self.tau =tau

	def testdataread(self):
		testdata = sio.loadmat(self.filename)
		# print(testdata)
		X_orig = testdata["X"]
		Y_orig = testdata["Y"]
		Z_orig = testdata["Z"]
		T_orig = testdata["times"]
		X = []
		Y = []
		Z = []
		T_acc = []
		T = []
		for i in X_orig:
			X.append(i[0])
		for i in Y_orig:
			Y.append(i[0])
		for i in Z_orig:
			Z.append(i[0])
		for i in T_orig:
			T_acc.append(i[0])
		T_arr = np.linspace(0,10,len(T_acc))
		for i in T_arr:
			T.append(i)
		dt = 10/len(T_acc)
		return X,Y,Z,T,dt

	def loadtestdata(self):
		self.X, self.Y, self.Z, self.T, self.dt = self.testdataread()
		return self.X,self.Y,self.Z,self.T,self.dt



if __name__ == "__main__":
	pj = ZHYDMPs(filename='./test.mat',alpha=10.0,beta=10.0,ax = 1.0,tau=10)
	x,y,z,t,dt = pj.loadtestdata()
	print(dt)
