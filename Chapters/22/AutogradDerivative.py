import jax


def cumulate_exponents(inputs):

    # Sigmoid Activation Funciton
    # f(x) = 1.0/(1.0 + e⁻ˣ)
    sigmoids = 1. / (1. + jax.numpy.exp(-inputs))
    #
    # Σsigmoids
    return jax.numpy.sum(sigmoids)

def test():
    
    # f'(x) = e⁻ˣ / (1 + e⁻ˣ)² = (1 - f(x)) x f(x)
    # derivative_cumulate_exponents = f'(x)
    # [0.25, ]
    derivative_cumulate_exponents = jax.grad(cumulate_exponents)
    
    # inputs = [0. 1. 2.]
    # sigmoids = [0.5, 0.731058578630005, 0.880797077977882]
    inputs = jax.numpy.arange(3.)
    
    # Σsigmoids = Σ[0.5, 0.731058578630005, 0.880797077977882] = 2.1118555
    outputs = cumulate_exponents(inputs)
    derivative_outputs = derivative_cumulate_exponents(inputs)
    
    print("inputs =", inputs, ", outputs =", outputs, ", derivative_outputs =", derivative_outputs)
    
if __name__ == "__main__":
    
    test()
