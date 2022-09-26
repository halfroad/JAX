import jax
import jax.numpy as jnp

def convolve():

    key = jax.random.PRNGKey(17)
    inputs = jnp.linspace(0, 9, 10)

    print("Inputs: ", inputs)

    kernel = jnp.ones(3) / 10.
    print("kernel: ", kernel)

    convoled = jnp.convolve(inputs, kernel, mode = "same")

    print("Convolved: ", convoled)

    convoled = jnp.convolve(inputs, kernel)

    print("Convolved: ", convoled)

def main():

    convolve()

if __name__ == "__main__":

    main()
