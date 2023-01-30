import jax
from sklearn.datasets import load_iris

def load():
    
    iris = load_iris()
    
    data = jax.numpy.float32(iris.data)
    targets = jax.numpy.float32(iris.target)
    
    return (data, targets)

def Dense(inputs_shape = (4, 1)):

    def init_function(shape = inputs_shape):
    
        key = jax.random.PRNGKey(10)
        
        weights, biases = jax.random.normal(key = key, shape = shape), jax.random.normal(key = key, shape = (shape[-1],))
        
        return (weights, biases)
        
    def apply_function(inputs, params):
    
        weights, biases = params
        
        dotted = jax.numpy.dot(inputs, weights) + biases
        
        return dotted
        
    return init_function, apply_function
    
def loss_function(params, inputs, genuineness, predict_function):

    """
    
    Loss Function: Mean Squared Error, g(x) = (f(x) - y)²
    
    """
    
    predictions = predict_function(inputs, params)
    averages = jax.numpy.mean(jax.numpy.power(predictions - genuineness, 2.0))
    
    return averages
    
def train():

    init_function, apply_function = Dense()
    
    params = init_function()
    learning_rate = 5e-3    # Learning rate
    epochs = 1000           # Number of iteration
    
    (data, targets) = load()
    
    for i in range(epochs):
    
        # Compute the losses
        loss = loss_function(params, data, targets, apply_function)
        
        if (i + 1) % 100 == 0:
        
            print(f" i = {i + 1}, loss: {loss}")
            
        # Compute the gradiences
        grad_loss_function = jax.grad(loss_function)
        gradients = grad_loss_function(params, data, targets, apply_function)
        
        params = [
            # Add the learning rate x (-derivative) for each parameter
            (param - gradient * learning_rate) for param, gradient in zip(params, gradients)
        ]
    
    
def test():

    train()
    
if __name__ == "__main__":

    test()
