import numpy as np

y_des = np.load('path1.npz')['A']

y_des = np.transpose(y_des)

print(len(y_des[0,:]))

y_des = y_des[:,1:]

print(y_des[:,69])
print(y_des[:,70])

numlist = []

for i in range(len(y_des[0,:])-1):
	a = y_des[:,i]
	b = y_des[:,i+1]
	c = np.linalg.norm(b-a)
	if c > 1e-3:
		numlist.append(i)
		#print(c)

print(numlist)
gen_arr = y_des[:,numlist]


gen_arr[0,:] = np.linspace(0,3,len(gen_arr[0,:]))
##### from datagen

# gen_arr[2,:] = np.sin(np.arange(0, 2, .01)*5) #np.linspace(0, 3, 200)
# gen_arr[0,:] = np.zeros(200)#np.linspace(0, 3, 200)#np.zeros(200)
# gen_arr[1,:] = np.zeros(200) #np.linspace(0, 3, 200)


#print(gen_arr)
np.savez_compressed('./path1_mod1',A = gen_arr)