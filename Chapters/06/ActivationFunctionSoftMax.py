import jax

def softmax(inputs, axis = -1):

    # jax.numpy.exp(): f(x) = eˣ
    unnormalized = jax.numpy.exp(inputs)
    accumulated = unnormalized.sum(axis, keepdims = True)
    
    probabilities = unnormalized / accumulated
    
    return probabilities
    
def test():

    array = jax.numpy.array([8, 5, 0])
    
    probabilities = softmax(array)
    
    jax.numpy.set_printoptions(suppress=True)
    
    print(f"{probabilities}")
    
    probabilities = jax.nn.softmax(array)
    
    print(f"{probabilities}")
    
if __name__ == "__main__":

    test()
