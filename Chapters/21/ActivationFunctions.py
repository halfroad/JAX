import jax

def sigmoid(x):
    
    return 1. / (1. + jax.numpy.exp(-x))

def sigmoid_derivative(x):
    
    return x * (1 - x)

def test():
    
    for i in range(-20, 20):
    
        print(sigmoid(i))
    
if __name__ == "__main__":
    
    test()