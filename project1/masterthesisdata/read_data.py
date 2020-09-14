import numpy as np

#### import data
a_list = np.load('joypath2.npz')['A']
print(a_list.shape)
b_list = np.load('joypath2.npz')['B']
print(b_list.shape)
##### find the data length
joy_len = a_list.shape[0]
print(joy_len)

pos_list = np.zeros((joy_len+1,3))
ori_list = np.zeros((joy_len+1,3))

#print(pos_list.shape)
#print(ori_list.shape)
#######
#pos_list[i,0] = a_list[]
#pos_list[i,1] = 
#pos_list[i,2] =

#ori_list[i,0] = 
#ori_list[i,1] =
#ori_list[i,2] = 
for i in range(1,joy_len+1):
	pos_list[i,0] = pos_list[i-1,0] + a_list[i-1,0]*0.03
	pos_list[i,1] = pos_list[i-1,1] + a_list[i-1,1]*0.03
	pos_list[i,2] = pos_list[i-1,2] + (a_list[i-1,2]-1)*(b_list[i-1,4]-0.5)*0.03

	ori_list[i,0] = ori_list[i-1,0] + a_list[i-1,3]*0.1
	ori_list[i,1] = ori_list[i-1,1] + a_list[i-1,4]*0.1
	ori_list[i,2] = ori_list[i-1,2] + (a_list[i-1,5]-1)*(b_list[i-1,5]-0.5)*0.01

### save
np.savez_compressed('./joy_gen4',A = pos_list,B = ori_list)