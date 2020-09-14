from contcode import VrepEnvBase
import vrep
import numpy as np

class MyClass(VrepEnvBase):

	def __init__(self):
		super(MyClass, self).__init__()

		rc, self.sphere_handle = vrep.simxGetObjectHandle(self.clientID, 'Baxter_target_dummy', vrep.simx_opmode_blocking)

		#rc, h_pos=vrep.simxGetObjectPosition(self.clientID, self.sphere_handle, -1,  vrep.simx_opmode_streaming)
		#rc, pos_init = vrep.simxGetObjectPosition(self.clientID, self.sphere_handle, -1,  vrep.simx_opmode_streaming

		rc1, self.y_pos = vrep.simxGetObjectPosition(self.clientID, self.sphere_handle, -1, vrep.simx_opmode_streaming)
		rc2, self.y_ori = vrep.simxGetObjectOrientation(self.clientID, self.sphere_handle, -1, vrep.simx_opmode_streaming)

		# self.y_pos = [0.0,0.0,0.0]
		# self.y_ori = [0.0,0.0,0.0]


	def run(self):

		rc1, self.y_pos = vrep.simxGetObjectPosition(self.clientID, self.sphere_handle, -1, vrep.simx_opmode_blocking)
		rc2, self.y_ori = vrep.simxGetObjectOrientation(self.clientID, self.sphere_handle, -1, vrep.simx_opmode_blocking)
		self.synchronous_trigger()

if __name__ == "__main__":
	# global pos_list, ori_list
	# pos_list = []
	# ori_list = []
	my_class = MyClass()
	file1=open("pos_genbax_4.txt","w")
	file2=open("ori_genbax_4.txt","w")

	while True:
		# try:
		# 	my_class.run()
		# 	# A = np.zeros((1,3))
		# 	# A[0,0] = B[0]
		# 	# A[0,1] = B[1]
		# 	# A[0,2] = B[2]
		# 	# print(A)
		# 	# out_mtx = np.r_[out_mtx,A]
		# 	# np.savez_compressed('./path2',A=out_mtx)
		# except SystemExit():
		# 	print('1')
		# 	pass
		my_class.run()
		# #print(a)
		# pos_list.append(pos)
		# ori_list.append(ori)
		#pos_list.append([my_class.y_pos])

		#ori_list.append(my_class.y_ori)
		#print(len(pos_list))
		#print(pos_list)
		print("1")
		#print(my_class.y_pos)
		#str1 = '%f %f %f\n'%()
		#file1.write(str(my_class.y_pos))
		file1.write("%f %f %f\n"%(my_class.y_pos[0],my_class.y_pos[1],my_class.y_pos[2]))
		#file1.write("\n")
		#file2.write(str(my_class.y_ori))
		file2.write("%f %f %f\n"%(my_class.y_ori[0],my_class.y_ori[1],my_class.y_ori[2]))
		#file2.write("\n")
		# if is_shutdown():
		#  	#np.savez_compressed('./joypath_baxter_4',A=pos_list,B=ori_list)
		#  	file1.close()
		#  	file2.close()
		#  	break

	#rospy.spin()