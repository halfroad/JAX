import jax
import time

def setup():
    
    prng = jax.random.PRNGKey(15)
    weight = 0.313
    bias = 1.526
    
    inputs = jax.random.normal(key = prng, shape = (10000,))
    genuines = weight * inputs + bias
    
    return ((prng, weight, bias), (inputs, genuines))

# Create model
def model(params, inputs):
    
    # Here the weight and bias are the params to be fitted.
    (weight, bias) = (params[0], params[1])
    
    # Compute the predictions relies on the current params
    predictions = inputs * weight + bias
    
    return predictions

# Loss function Mean Squared Error to compute the losses
def loss_function(params, inputs, genuines):
    
    predictions = model(params, inputs)
    
    # mse = 1/n(y - f(x))²
    meanSquaredError = jax.numpy.mean((predictions - genuines) ** 2)
    
    return meanSquaredError

# Update function to update the params by Gradient Descend
def update_function(params, inputs, genuines, learning_rate = 1e-3):
    
    loss_function_grad = jax.grad(loss_function)
    gradient = loss_function_grad(params, inputs, genuines)
    
    # Update the current params - weight and bias
    params = params - learning_rate * gradient
    
    return params

def train():
    
    ((prng, weight, bias), (inputs, genuines)) = setup()
    print(((prng, weight, bias), (inputs.shape, genuines.shape)))
    
    # Initial params could be huge difference from real params,
    # but don't worry, the params will be fitted iterations, and
    # become closer and closer to the real params
    params = jax.random.normal(key = prng, shape = (2,))
    print("Initial params = ", params)
    
    start = time.time()
    grandStart = start
    epochs = 100000
    
    for i in range(epochs):
        
        if (i  + 1) % 100 == 0:
            
            loss = loss_function(params, inputs, genuines)
            params = update_function(params, inputs, genuines)
            end = time.time()
            
            print(f"Loss = {loss},", f"weight = {params[0]}, bias = {params[1]}", f"Time consumed: %.12f seconds" % (end -start), f"while iterating epoch {i + 1}")
            
            start = time.time()
            
    print("Time consumed: %.12f seconds" % (end - grandStart), f"while iterating {epochs} epochs")
    
if __name__ == "__main__":
    
    train()