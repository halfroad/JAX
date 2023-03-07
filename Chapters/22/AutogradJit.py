import jax
import time

def cumulate_exponents1(inputs):
    
    return jax.numpy.sum(1. / (1. + jax.numpy.exp(-inputs)))

def cumulate_exponents2(inputs):
    
    return jax.numpy.sum(1. / (1. + jax.numpy.exp(-inputs)))

def cumulate_exponents3(inputs):
    
    return jax.numpy.sum(1. / (1. + jax.numpy.exp(-inputs)))

@jax.jit
def cumulate_exponents_jit(inputs):
    
    return jax.numpy.sum(1. / (1. + jax.numpy.exp(-inputs)))

def test():
    
    inputs = jax.numpy.arange(1024.)
    
    start = time.time()
    
    cumulate_exponents_jit_derivative = jax.grad(cumulate_exponents_jit)
    
    print("cumulate_exponents_jit_derivative(inputs) = ", cumulate_exponents_jit_derivative(inputs))
    
    end = time.time()
    
    print("Time consumed: %.2f" % (end - start))
    print("------------------------------")
    
    start = time.time()
    
    cumulate_exponents1_derivative = jax.grad(cumulate_exponents1)
    
    print("cumulate_exponents1_derivative(inputs) = ", cumulate_exponents1_derivative(inputs))

    end = time.time()
    
    print("Time consumed: %.2f" % (end - start))
    print("------------------------------")
    
    start = time.time()
    
    cumulate_exponents2_derivative = jax.grad(cumulate_exponents2)
    cumulate_exponents2_derivative_jit = jax.jit(cumulate_exponents2_derivative)
    
    print("cumulate_exponents2_derivative_jit(inputs) = ", cumulate_exponents2_derivative_jit(inputs))

    end = time.time()
    
    print("Time consumed: %.2f" % (end - start))
    
if __name__ == "__main__":
    
    test()
    
    
    
    
    
    
          
    