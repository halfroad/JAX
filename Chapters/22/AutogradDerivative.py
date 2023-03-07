import jax


def cumulate_exponents(inputs):
    
    return jax.numpy.sum(1. / (1. + jax.numpy.exp(-inputs)))

def test():
    
    derivative_cumulate_exponents = jax.grad(cumulate_exponents)
    
    inputs = jax.numpy.arange(3.)
    
    derivative_outputs = derivative_cumulate_exponents(inputs)
    
    print("inputs =", inputs, ", derivative_outputs =", derivative_outputs)
    
if __name__ == "__main__":
    
    test()