import jax

def convolve_one_dimension() -> jax.numpy.array:

    inputs = jax.numpy.linspace(0, 9, 10)
    kernel = jax.numpy.ones(3) / 10
    
    outputs = jax.numpy.convolve(inputs, kernel, mode = "same")
    
    print(f"inputs = {inputs}, kernel = {kernel}, outputs = {outputs}")
    
def test():

    convolve_one_dimension()

if __name__ == "__main__":
    
    test()
             
             
