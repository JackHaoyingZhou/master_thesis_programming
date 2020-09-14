from contcode import VrepEnvBase
import vrep
import numpy as np

class MyClass(VrepEnvBase):

	def __init__(self):
		super(MyClass, self).__init__()

		rc, self.sphere_handle = vrep.simxGetObjectHandle(self.clientID, 'Sphere', vrep.simx_opmode_blocking)

		vrep.simxGetObjectPosition(self.clientID, self.sphere_handle, -1,  vrep.simx_opmode_streaming)

	def set_gripper_state(self, gripper_state):
		if gripper_state < 0.5:
			vrep.simxSetIntegerSignal(self.clientID,'baxter_open_gripper',0,vrep.simx_opmode_oneshot)
			self.gripper_state = 0
		else:
			vrep.simxSetIntegerSignal(self.clientID,'baxter_open_gripper',1,vrep.simx_opmode_oneshot)
			self.gripper_state = 1

		self.synchronous_trigger()

if __name__ == "__main__":
	my_class = MyClass()
	g_state1 = np.linspace(0,1,50)
	g_state2 = np.linspace(1,0,50)
	g_state = np.hstack((g_state1,g_state2))
	#print(g_state)
	for i in range(100):
		my_class.set_gripper_state(g_state[i])
		print(i)

