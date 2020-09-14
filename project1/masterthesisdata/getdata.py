import rospy
from sensor_msgs.msg import Joy
import numpy as np

global axes_list, buttons_list
axes_list = []
buttons_list = []
def callback(data):
	axes_list.append(data.axes)
	buttons_list.append(data.buttons)
	np.savez_compressed('./joypath3',A=axes_list,B=buttons_list)



def start():
	rospy.init_node('getdata',anonymous = True)
	rospy.Subscriber('joy',Joy,callback)

	rospy.spin()


if __name__ == '__main__':
	try:
		start()
	except rospy.ROSInterruptException:
		pass
