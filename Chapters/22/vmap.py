import jax
import time

def cumulate_exponents(inputs):
    
    return jax.numpy.sum(1. / (1. + jax.numpy.exp(-inputs)))

def test():
    
    start = time.time()
    inputs = jax.numpy.arange(1024000.)
    
    cumulate_exponents1_derivative = jax.grad(cumulate_exponents)
    cumulate_exponents1_derivative(inputs)
    
    end = time.time()
    
    print("Time consumed: %.2f" % (end - start), "when function executing cumulate_exponents1_derivative")
    
    start = time.time()
    inputs = jax.numpy.arange(1024000.)
    
    cumulate_exponents2_derivative = jax.vmap(jax.grad(cumulate_exponents))
    cumulate_exponents2_derivative(inputs)
    
    end = time.time()
    
    print("Time consumed: %.2f" % (end - start), "when function executing cumulate_exponents2_derivative")

    start = time.time()
    inputs = jax.numpy.arange(1024000.)
    
    cumulate_exponents3_derivative = jax.jit(jax.vmap(jax.grad(cumulate_exponents)))
    
    cumulate_exponents3_derivative(inputs)
    
    end = time.time()
    
    print("Time consumed: %.2f" % (end - start), "when function executing cumulate_exponents3_derivative")

    
if __name__ == "__main__":
    
    test()