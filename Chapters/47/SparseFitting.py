import jax
from jax.experimental import sparse

def setup():

    key = jax.random.PRNGKey(15)
    number_classes = 10
    classes = jax.numpy.arange(number_classes)
    
    inputs = []
    genuines = []
    
    for i in range(1024):
    
        x = jax.random.choice(key + 1, classes, shape = (1,))
        x = x[0]
        x_one_hoted = jax.nn.one_hot(x, num_classes = number_classes)
        
        inputs.append(x_one_hoted)
        genuines.append(x)
        
    params = [jax.random.normal(key = key, shape = (number_classes, 1)), jax.random.normal(key = key, shape = (1,))]
    
    sparsed_inputs = sparse.BCOO.fromdense(jax.numpy.array(inputs))
    genuines = jax.numpy.array(genuines)
    
    epochs = 10000
    learning_rate = 1e-3
    
    return (key, number_classes, params, epochs, learning_rate), (inputs, sparsed_inputs, genuines)
    
# Activation function
def sigmoid(inputs):

    return 1 / 2 * (jax.numpy.tanh(inputs / 2) + 1)
    
# Prediction model
def predict(params, inputs):

    predictions = jax.numpy.dot(inputs, params[0]) + params[1]
    
    return sigmoid(predictions)

def loss_function(params, sparsed_inputs, genuines):

    sparsed_predict = sparse.sparsify(predict)
    predictions = sparsed_predict(params, sparsed_inputs)
    
    losses = genuines * jax.numpy.log(predictions) + (1 - genuines) * jax.numpy.log(1 - predictions)
    losses = -jax.numpy.mean(losses)
    
    return losses
    
def train():

    (key, number_classes, params, epochs, learning_rate), (inputs, sparsed_inputs, genuines) = setup()

    losses = loss_function(params, sparsed_inputs, genuines)
    
    print("losses post the train = ", losses)
    print("params prior to the train = ", params)
    
    for i in range(epochs):
    
        loss_function_grad = jax.grad(loss_function)
        
        gradients = loss_function_grad(params, sparsed_inputs, genuines)
        params = [param - gradient * learning_rate for param, gradient in zip(params, gradients)]
        
        if (i + 1) % 100 == 0:
        
            print(f"losses after {i + 1} = ", losses)
        
    losses = loss_function(params, sparsed_inputs, genuines)
    
    print("losses post the train = ", losses)

def main():

    train()
    
if __name__ == "__main__":

    main()
