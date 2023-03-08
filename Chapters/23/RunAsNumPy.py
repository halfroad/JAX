import jax

def run():
    
    array = jax.numpy.linspace(0, 9, 10)    
    print(array)
    

def matrix_multiplify():
    
    prng = jax.random.PRNGKey(15)
    
    matrix1 = jax.random.normal(key = prng, shape = (2, 3))
    matrix2 = jax.random.normal(key = prng, shape = (3, 1))
    
    result = jax.numpy.matmul(matrix1, matrix2)    
    print("matmul(matrix1, matrix2) =", result)
    
    result = jax.numpy.dot(matrix1, matrix2)
    print("dot(matrix1, matrix2) =", result)
    
def immutable_array():
    
    array = jax.numpy.linspace(0, 9, 10)
    
    print(type(array))

    array[5] = 10
    
    print("The new array =", array)
    
def update_array():
    
    array = jax.numpy.linspace(0, 9, 10)
    array_new = array.at[5].set(10)
    
    print("array =", array)
    print("array_new =", array_new)
    
if __name__ == "__main__":
    
    '''
    run()
    matrix_multiplify()
    immutable_array()
    '''
    update_array()