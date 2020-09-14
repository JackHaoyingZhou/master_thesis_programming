import numpy as np


gen_arr = np.ones((3,200));

gen_arr[2,:] = 0.4*np.sin(np.arange(0, 2, .01)*5) + 0.8*np.exp(np.arange(-2, 0, .01)) + 0.05*np.arange(0, 2, .01)**2 #np.linspace(0, 3, 200)
gen_arr[0,:] = 0.2*np.cos(np.arange(0, 2, .01)*4) + 0.9*np.exp(np.arange(-2, 0, .01)) + 0.04*np.arange(0, 2, .01)**2#np.linspace(0, 3, 200)#np.zeros(200)
gen_arr[1,:] = 0.3*np.sin(np.arange(0, 2, .01)*6) + 0.85*np.exp(np.arange(-2, 0, .01)) + 0.06*np.arange(0, 2, .01)**2#np.linspace(0, 3, 200)
# gen_arr[3,:] = np.sin(np.arange(0, 2, .01)*5)
# gen_arr[4,:] = np.cos(np.arange(0, 2, .01)*5)
# gen_arr[5,:] = np.cos(np.arange(0, 2, .01)*5)
#gen_arr[2,:] = np.arange(0, 2, .01)**2

#print(gen_arr)
np.savez_compressed('./gen_path21',A = gen_arr)
