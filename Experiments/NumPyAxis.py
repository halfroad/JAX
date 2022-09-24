import numpy as np

a = np.arange(24).reshape(2, 3, 4)

print("a = ", a)

b = np.sum(a, axis = 0)
print("np.sum(a, axis = 0) = ", b)

c = np.sum(a, axis = 1)
print("np.sum(a, axis = 1)= ", c)

d = np.sum(a, axis = 2)
print("np.sum(a, axis = 2) = ", d)
