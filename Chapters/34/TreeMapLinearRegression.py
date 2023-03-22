import jax
import time

def setup():

    prng = jax.random.PRNGKey(15)
    
    inputs = jax.random.normal(key = prng, shape = (1000, 1))
    
    weight = 1.111
    bias = 0.618
    
    genuines = weight * inputs + bias
    
    layers_shape = [1, 64, 128, 1]
    epochs = 4000
    params = init_mlp_params(prng, layers_shape)
    
    return (prng, (weight, bias, layers_shape, epochs, params), (inputs, genuines))
    
def init_mlp_params(prng, layers_shape):

    params = []
    
    # zip([1, 64, 128], [64, 128, 1]) -> [1, 64], [64, 128], [128, 1]
    for _in, _out in zip(layers_shape[: -1], layers_shape[1:]):
    
        weights = jax.random.normal(key = prng, shape = (_in, _out)) / 128.
        biases = jax.random.normal(key = prng, shape = (_out,)) / 128.
        
        _dict = dict(weights = weights, biases = biases)
        
        params.append(_dict)
        
    return params

# Computing Function of multilayers
@jax.jit
def forward(params, inputs):

    for param in params:
    
        inputs = jax.numpy.matmul(inputs, param["weights"]) + param["biases"]
        
    return inputs
    
# Loss Funciton
@jax.jit
def loss_function(params, inputs, genuines):

    predictions = forward(params, inputs)
    losses = (genuines - predictions) ** 2
    losses = jax.numpy.mean(losses)
    
    return losses
    
# Optimizer Function: Stochastic Gradient Descent
@jax.jit
def optmizer_function(params, inputs, genuines, learning_rate = 1e-3):

    loss_function_grad = jax.grad(loss_function)
    gradients = loss_function_grad(params, inputs, genuines)
    
    params = params - gradients * learning_rate
    
    return params

# Work but not recommanded
@jax.jit
def optmizer_function_v2(params, inputs, genuines, learning_rate = 1e-3):

    loss_function_grad = jax.grad(loss_function)
    gradients = loss_function_grad(params, inputs, genuines)
    
    _params = []
    
    for param, gradient in zip(params, gradients):
    
        weights = param["weights"] - learning_rate * gradient["weights"]
        biases = param["biases"] - learning_rate * gradient["biases"]
        
        _dict = dict(weights = weights, biases = biases)
        
        _params.append(_dict)
        
    return _params

# Recommanded
@jax.jit
def optmizer_function_v3(params, inputs, genuines, learning_rate = 1e-1):

    loss_function_grad = jax.grad(loss_function)
    gradients = loss_function_grad(params, inputs, genuines)
    
    params = jax.tree_util.tree_map(lambda param, gradient: param - learning_rate * gradient, params, gradients)
    
    return params
    
def train():

    (prng, (weight, bias, layers_shape, epochs, params), (inputs, genuines)) = setup()
    
    start = time.time()
    
    for i in range(epochs):
    
        params = optmizer_function_v3(params, inputs, genuines)
        
        if (i + 1) % 100 == 0:

            losses = loss_function(params, inputs, genuines)
            end = time.time()

            print(f"Loss now is {losses}, time consumed {end - start} seconds, iteration is {i + 1}")

            start = time.time()

    tests = jax.numpy.array([2.3])

    print("Genuine = ", tests * weight + bias)
    print("Prediction = ", forward(params, tests))
    
def test():

    train()
    
if __name__ == "__main__":
    
    test()
