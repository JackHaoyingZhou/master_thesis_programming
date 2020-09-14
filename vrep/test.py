from contcode import VrepEnvBase
import vrep
import numpy as np

class MyClass(VrepEnvBase):

	def __init__(self):
		super(MyClass, self).__init__()

		rc, self.sphere_handle = vrep.simxGetObjectHandle(self.clientID, 'Sphere', vrep.simx_opmode_blocking)

		vrep.simxGetObjectPosition(self.clientID, self.sphere_handle, -1,  vrep.simx_opmode_streaming)


	def run(self):
		rc, sphere_positions = vrep.simxGetObjectPosition(self.clientID, self.sphere_handle, -1, vrep.simx_opmode_buffer)
		self.synchronous_trigger()
		print(sphere_positions)
		print(sphere_positions[1])
		return sphere_positions

if __name__ == "__main__":
	my_class = MyClass()
	
	out_mtx = np.zeros((1,3))

	while True:
		B = my_class.run()
		A = np.zeros((1,3))
		A[0,0] = B[0]
		A[0,1] = B[1]
		A[0,2] = B[2]
		print(A)
		#print(out_mtx.shape)
		out_mtx = np.r_[out_mtx,A]
		np.savez_compressed('./path3',A=out_mtx)
		#time.sleep(0.5)
