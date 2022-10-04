import numpy as np

array = [[0, 1, 2],
         [2, 1, 3]]

a = np.sum(array)

print(a)

b = np.sum(array, axis = 0)

print(b)

c = np.sum(array, axis = 1)

print(c)

array = np.array([[0, 2, 1]])

print(array.sum())
print(array.sum(axis = 0))
print(array.sum(axis = 1))

array = np.array([0, 2, 1])

print(array.sum())
print(array.sum(axis = 0))
# print(array.sum(axis = 1))

array = [0, 1, 2, 3, 4, 5, 6]
print(array[-1])
