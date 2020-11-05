import numpy as np

print(np.arange(10))

"""
Creating a NumPy Array
"""

# Basic multi-dimensional array
data = [1, 2, 3, 4, 5]  # python list
print(data)
print(type(data))

data = np.array(data)  # numpy array
print(data)
print(type(data))

print(data.dtype)  # check the current datatype

data = np.array(data, dtype=np.float64)
print(data)
print(data.dtype)  # checking the datatype

data = np.array(data, dtype=np.int64)
print(data)
print(data.dtype)  # checking the data type

# Array of zeros
data = np.zeros((1, 5))
print(data)

data = np.zeros((2, 2))
print(data)

# Array of ones
data = np.ones((3, 3))
print(data)

# Random numbers in ndarrays
data = np.random.rand(3, 3)
print(data)

# An array of your choice
data = np.full((2, 2), 5)  # create a 2x2 matrix and full the array with 5's
print(data)

# Identity matrix in NumPy
data = np.eye(3, k=0)
print(data)

data = np.eye(3, k=1)  # we can change the diagonal axis by modifying the k argument
print(data)

# Evenly Spaced n-Dimensional Array
data = np.arange(10)  # # print from number 0 - 9 sequentially
print(data)

data = np.arange(2, 12, 2)  # # print the first 5 even numbers
print(data)

data = np.linspace(0, 10, 5)
print(data)

"""
Checking the Shape of NumPy Array
"""

# Dimensions of NumPy arrays
data = np.array([[2, 3, 4],
                 [5, 6, 7],
                 [8, 9, 10]])

print(f"Data: {data}\n")  # view the array
print(f"Number of dimensions: {data.ndim}")  # check the number of dimensions

# Shape of NumPy array
print(f"Shape: {data.shape} \n")  # 3x3 matrix (array)
print(f"Rows: {data.shape[0]} \n")  # get the number of rows
print(f"Columns: {data.shape[1]} \n")  # get the number of columns

# Reshaping the NumPy Array
data = np.arange(9)
print(f"Data: {data}")
print(f"Shape: {data.shape}")

data = np.reshape(data, (3, 3))  # reshape into a 3x3 matrix
print(f"Data: {data}")
print(f"Shape: {data.shape}")

data = np.reshape(data, (3, -1))  # reshape into a 3x3 matrix
print(data)

data = np.reshape(data, (-1, 3))  # reshape into a 3x3 matrix
print(data)

# Flattening a NumPy array
data = np.random.rand(2, 2)
print(data)

print(f"Flatten: {data.flatten()}")
print(f"Ravel: {data.ravel()}")

data = np.zeros((2, 2))  # create a 2x2 array
flatten = data.flatten()
flatten[0] = 1000  # modified the first value to 1000
print(f"Data: {data}")

ravel = data.ravel()
ravel[0] = 500  # modified the first value to 500
print(f"Data: {data}")

# Transpose of a NumPy Array
data = np.array([[1, 2, 3],
                 [4, 5, 6]])
print(f"Data: {data}\n")
print(f"Shape: {data.shape}")

data = np.transpose(data)
print(f"Data: {data}\n")
print(f"Shape: {data.shape}")

"""
Indexing with NumPy Array
"""
# One-dimensional subarrays
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(data)
print(data[0])
print(data[4])
print(data[-1])

data[0] = 200.55
print(data)

# Two-dimensional subarrays
data = np.array([[1, 2],
                 [4, 5]])
print(data)

print(data[0, 0])

data[0, 0] = 500
print(data)

"""
Slicing of NumPy Arrays
"""
# One-dimensional subarrays
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(data)
print(data[1:5])
print(data[::2])
print(data[::-1])

# Two-dimensional subarrays
data = np.array([[1, 2, 3],
                 [4, 5, 6]])
print(data)
print(data[0:2, ::2])  # or print(data[-1, -1])
print(data[:, 0])
print(data[-1, :])  # or print(data[1, :])

data = np.array([[[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]]])

# Three-dimensional sub-arrays
print(f"Data: {data}")
print(f"Shape: {data.shape}")

"""
Array Concatenation
"""

a = np.arange(3)
b = np.arange(3, 6)

print(f"a: {a}")
print(f"b: {b}")

# np.vstack
data = np.vstack([a, b])
print(f"Data: {data}")
print(f"Shape: {data.shape}")

# np.hstack
data = np.hstack([a, b])
print(f"Data: {data}")
print(f"Shape: {data.shape}")

# np.concatenate
a = np.array([['a', 'b', 'c']]).T
b = np.array([['d', 'd', 'e']]).T

print(f"a: {a}")
print(f"b: {b}")

data = np.concatenate([a, b], axis=0)  # concatenate row-wise
print(f"Data: {data}")

data = np.concatenate([a, b], axis=1)  # concatenate column-wise
print(f"Data: {data}")

"""
Broadcasting in NumPy Array
"""
a = np.full((4, 4), 5)
b = np.full(1, 1)
print(f"a: {a}")
print(f"b: {b}")

data = a + b
print(f"Data: {data}")

"""
Mathematical function with NumPy Array
"""

# Array Arithmetic with NumPy
data = np.array([4, 4, 4, 4])
print(f"Data: {data}")

print(f"Power: {np.power(data, 2)}")
print(f"Remainder: {np.mod(data, 2)}")
print(f"Divide: {np.divide(data, 2)}")
print(f"Multiply: {np.multiply(data, 2)}")
print(f"Addition: {np.add(data, 2)}")
print(f"Substraction: {np.subtract(data, 2)}")

# Quartile, Mean, Median and Standard deviation
data = np.array([2, 4, 6, 7, 10])
print(f"Data: {data}")

print(f"Median: {np.median(data)}")
print(f"Mean: {np.mean(data)}")
print(f"Standard Deviation: {np.std(data)}")

print(f"Minimum: {np.min(data)}")
print(f"Maximum: {np.max(data)}")

print(f"25th percentile: {np.percentile(data, 25)}")
print(f"Median : {np.median(data)}")
print(f"75th percentile: {np.percentile(data, 75)}")

min_index = np.argmin(data)  # get the index of the minimum value
max_index = np.argmax(data)  # get the index of the maximum value

print(f"Minimum Index: {min_index} \t Value: {data[min_index]}")
print(f"Maximum Index: {max_index} \t Value: {data[max_index]}")

# Sorting in NumPy arrays
data = np.array([10, 2, 8, 4, 6, 5])
print(f"Before sorting: {data}")

data = np.sort(data)  # sort the array
print(f"After sorting: {data}")
