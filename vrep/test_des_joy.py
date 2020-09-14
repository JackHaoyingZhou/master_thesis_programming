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

	def __init__(self):
		super(MyClass, self).__init__()

		rc, self.sphere_handle = vrep.simxGetObjectHandle(self.clientID, 'Baxter_target_dummy', vrep.simx_opmode_blocking)

		vrep.simxGetObjectPosition(self.clientID, self.sphere_handle, -1,  vrep.simx_opmode_streaming)


	def run(self, y_pos,y_ori):
		self.y_pos = y_pos
		self.y_ori = y_ori
		vrep.simxSetObjectPosition(self.clientID, self.sphere_handle, -1, self.y_pos, vrep.simx_opmode_blocking)
		vrep.simxSetObjectOrientation(self.clientID, self.sphere_handle, -1, self.y_ori, vrep.simx_opmode_blocking)
		self.synchronous_trigger()

if __name__ == "__main__":

	# pos = np.loadtxt('pos_multi_4.txt')
	# ori = np.loadtxt('ori_multi_4.txt')

	# pos = np.loadtxt('pos_4.txt')
	# ori = np.loadtxt('ori_4.txt')

	# pos = np.loadtxt('pos_genbax_3.txt')
	# ori = np.loadtxt('ori_genbax_4.txt')

	# pos = np.loadtxt('pos_s_3.txt')
	# ori = np.loadtxt('ori_s_3.txt')

	pos = np.load('complex_2_pos.npz')['A']
	ori = np.load('complex_2_ori.npz')['A']

	# pos = np.load('simple_1_pos.npz')['A']
	# ori = np.load('simple_1_ori.npz')['A']

	# pos = np.load('simple_m0_pos.npz')['A']
	# ori = np.load('simple_m4_ori.npz')['A']

	# pos = np.load('complex_1_pos.npz')['A']
	# # pos = np.load('test_5_pos.npz')['A']
	# ori = np.load('complex_1_ori.npz')['A']

	# pos = np.load('complex_mod_pos.npz')['A']
	# # pos = np.load('test_5_pos.npz')['A']
	# ori = np.load('complex_mod_ori.npz')['A']

	# pos = np.load('PC_2_pos.npz')['A']
	# ori = np.load('PC_2_ori.npz')['A']

	print(pos.shape)
	print(ori.shape)

	pos = np.transpose(pos)
	ori = np.transpose(ori)

	# pos = np.loadtxt('pos_genbax_3.txt')
	# ori = np.loadtxt('ori_genbax_4.txt')

	my_class = MyClass()

	for i in range(pos.shape[0]):
		my_class.run(y_pos=pos[i,:],y_ori=ori[i,:])