import jax

matrix = jax.numpy.array([
    [1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.]
])
print(matrix.shape)

addent = jax.numpy.array([
    [2],
    [3],
    [4]
])


print(addent.shape)
result = matrix + addent + 1 + 9

print(result)
print(result.shape)
