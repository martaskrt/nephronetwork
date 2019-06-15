import numpy as np

num_permutations = 1000
perm_length = 16

perm_arr = np.zeros((num_permutations, perm_length))
for i in range(num_permutations):
    arr = np.arange(perm_length)
    np.random.shuffle(arr)
    perm_arr[i, :] = arr

print(perm_arr)
file_handle = "permutations_" + str(num_permutations) + "_" + str(perm_length) + ".npy"
np.save(file_handle, perm_arr)