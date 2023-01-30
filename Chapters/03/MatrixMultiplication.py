import jax

def multiplify(matrix, weights, bias):
    
    result = jax.numpy.matmul(matrix, weights) + bias
    
    return result
    
if __name__ == "__main__":
    
    matrix = jax.numpy.array([[1.1, 1.8], [1.2, 1.7]])
    weights = jax.numpy.array([[3], [2]])
    bias = 0.4
    
    result = multiplify(matrix, weights, bias)
    
    print(result)
    
    
