from contcode import VrepEnvBase
import vrep
import rospy
from sensor_msgs.msg import Joy
import numpy as np

class Data_ext():
	def __init__(self):
		rospy.init_node('getdata',anonymous = True)
		self.axes_list = (0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
		self.buttons_list = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

	def callback(self,data):
		self.axes_list=(data.axes)
		self.buttons_list=(data.buttons)
		#print(axes_list)
		#print(buttons_list)
	
	def start(self):
		rospy.init_node('getdata',anonymous = True)
		rospy.Subscriber('joy',Joy,self.callback)
		rospy.sleep(0.1)
		#rospy.spin()
		#print(self.axes_list)
		return self.axes_list, self.buttons_list

class MyClass(VrepEnvBase):

	def __init__(self):
		super(MyClass, self).__init__()

		rc, self.sphere_handle = vrep.simxGetObjectHandle(self.clientID, 'Baxter_target_dummy', vrep.simx_opmode_blocking)

		rc, h_pos=vrep.simxGetObjectPosition(self.clientID, self.sphere_handle, -1,  vrep.simx_opmode_streaming)
		#rc, pos_init = vrep.simxGetObjectPosition(self.clientID, self.sphere_handle, -1,  vrep.simx_opmode_streaming)


		self.y_pos = [0.31604,-1.6456,0.62613]
		#self.y_ori = [179.82,-2.5873,-97.951]
		self.y_ori = [179.82/180.0*np.pi,-2.5873/180.0*np.pi,-97.951/180.0*np.pi]



	def run(self):
		ext_class = Data_ext()
		a,b = ext_class.start()
		#print(type(a))
		#print(a)
		#print(b)
		#print(a[2])
		#print(axes_list)
		#print(buttons_list)
		self.y_pos[0] = self.y_pos[0] + a[0]*0.01
		self.y_pos[1] = self.y_pos[1] + a[1]*0.01
		self.y_pos[2] = self.y_pos[2] + (a[2]-1)*(b[4]-0.5)*0.01
		self.y_ori[0] = self.y_ori[0] + a[3]*0.1
		self.y_ori[1] = self.y_ori[1] + a[4]*0.1
		self.y_ori[2] = self.y_ori[2] + (a[5]-1)*(b[5]-0.5)*0.1
		vrep.simxSetObjectPosition(self.clientID, self.sphere_handle, -1, self.y_pos,vrep.simx_opmode_blocking)
		vrep.simxSetObjectOrientation(self.clientID, self.sphere_handle, -1, self.y_ori,vrep.simx_opmode_blocking)
		self.synchronous_trigger()
		#print(self.y_pos)
		#print(sphere_positions)
		#print(sphere_positions[1])
		#pos_list.append(self.y_pos)
		#ori_list.append(self.y_ori)
		#print(pos_list)
		#np.savez_compressed('./joypath_baxter_5',A=pos_list,B=ori_list)
		#print(self.y_pos)
		#return self.y_pos, self.y_ori

if __name__ == "__main__":
	# global pos_list, ori_list
	# pos_list = []
	# ori_list = []
	my_class = MyClass()
	file1=open("pos_multi_1.txt","w")
	file2=open("ori_multi_1.txt","w")

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
		if rospy.is_shutdown():
		 	#np.savez_compressed('./joypath_baxter_4',A=pos_list,B=ori_list)
		 	file1.close()
		 	file2.close()
		 	break

	#rospy.spin()