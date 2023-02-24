import jax

def error_function(inputs, genuines, theta):
    
    # Least Square
    predictions = jax.numpy.dot(inputs, theta)
    transposed = jax.numpy.transpose(predictions)
    rows = inputs.shape[0]
    
    errors = 1. / (2 * rows) * jax.numpy.dot(transposed, predictions)
    
    return errors

def gradient_function(inputs, genuines, theta):
    
    predictions = jax.numpy.dot(inputs, theta) - genuines
    transposed = jax.numpy.transpose(inputs)
    rows = inputs.shape[0]
    
    gradient = 1. / rows * jax.numpy.dot(transposed, predictions)
    
    return gradient

def gradient_descent(inputs, genuines, alpha):
    
    theta = jax.numpy.array([1., 1.]).reshape(2, 1)
    gradient = gradient_function(inputs, genuines, theta)
    
    while not jax.numpy.all(jax.numpy.absolute(gradient) < 1e-5):
        
        theta = theta - alpha * gradient
        gradient = gradient_function(inputs, genuines, theta)
        
    return theta

def train():
    
    rows = 20
    
    # Createa a 2 dimensions matrix
    inputs0 = jax.numpy.ones((rows, 1))
    inputs1 = jax.numpy.arange(1, rows + 1).reshape(rows, 1)
    
    inputs = jax.numpy.hstack((inputs0, inputs1))
    genuines = jax.numpy.array([3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12, 11, 13, 13, 16, 17, 18, 17, 19, 21]).reshape(rows, 1)
    alpha = 1e-2
    
    theta = gradient_descent(inputs, genuines, alpha)
    
    print("Optimal:", theta)
    
    errors = error_function(inputs, genuines, theta)
    print("Error function:", errors)
    
if __name__ == "__main__":
    
    train()
