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

		rc, self.sphere_handle = vrep.simxGetObjectHandle(self.clientID, 'Sphere', vrep.simx_opmode_blocking)

		vrep.simxGetObjectPosition(self.clientID, self.sphere_handle, -1,  vrep.simx_opmode_streaming)


	def run(self, y_pos):


		vrep.simxSetObjectPosition(self.clientID, self.sphere_handle, -1, y_pos,vrep.simx_opmode_blocking)
		self.synchronous_trigger()

if __name__ == "__main__":

	y_des = np.load('joy_gen3.npz')['A']

	my_class = MyClass()

	for i in range(y_des.shape[0]):
		my_class.run(y_pos=y_des[i,:])