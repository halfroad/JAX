import jax.numpy as jnp

from jax import random

def vector_multiply():
    
    # Equals x = np.arange(0, 5)
    x = jnp.array([0, 1, 2, 3, 4])
    y = x[:: -1]

    print("x = ", x)
    print("x.shape = ", x.shape)
    
    print("y = ", y)
    print("y.shape = ", y.shape)

    z = jnp.dot(x, y)

    print("x * y = ", z)
    
def matrix_multiply():
    
    x = jnp.arange(0, 5)
    
    print("x = ", x)
    print("x.shape = ", format(x.shape))
    
    # keys = [key for i in range(5) for key in random.PRNGKey(i)]
    key = random.PRNGKey(5)
    y = random.randint(key, shape = (5, 1), minval = 0, maxval = 10)
    
    print("y = ", y)
    print("y.shape = ", format(y.shape))
    
    z = jnp.dot(x, y)
    
    print("jnp.dot(x, y) = ", format(z))
    
def matrix_multiply2():
    
    x = jnp.arange(0, 6).reshape(2, 3)
    
    print("x = ", x)
    print("x.shape = ", format(x.shape))
    
    key = random.PRNGKey(5)
    y = random.randint(key, shape = (3, 2), minval = 0, maxval = 10)
    
    print("y = ", y)
    print("y.shape = ", format(y.shape))
    
    z = jnp.dot(x, y)
    
    print("jnp.dot(x, y) = ", format(z))
    
def matrix_multiply3():
    
    array = jnp.array([[1.7, 1.7],
                       [2.14, 2.14]])
    weight = jnp.array([[1], [2]])
    bias = 0.99
    
    fx = jnp.matmul(array, weight) + bias
    
    print("fx = {}".format(fx))
    
def start():
    
    # vector_multiply()
    # matrix_multiply()
    # matrix_multiply2()
    matrix_multiply3()

start()
