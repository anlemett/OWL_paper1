# Import numpy
import numpy as np
import random

# Creating two numpy arrays
arr1 = np.array([[[1,2],[1,2]],[[3,4],[1,2]],[[9,10],[1,2]],[[11,12],[1,2]]])
arr2 = np.array([5,6,7,8])

print(arr1.shape)
print(arr2.shape)

# Display original arrays
print("Original Array 1:\n",arr1,"\n")
print("Original Array 2:\n",arr2,"\n")

# Combining both arrays to create a 2d array
#res = np.column_stack((arr1,arr2))
zipped = list(zip(arr1, arr2))

# Display the result
print("Result:\n",zipped)

random.shuffle(zipped)
arr1, arr2 = zip(*zipped)

# Display new arrays
print("New Array 1:\n",arr1,"\n")
print("New Array 2:\n",arr2,"\n")

