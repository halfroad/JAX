import jax
import jax.numpy as jnp

def convolve():

    key = jax.random.PRNGKey(17)
    inputs = jnp.linspace(0, 9, 10)

    print("Inputs: ", inputs)

    kernel = jnp.ones(3) / 10.
    print("kernel: ", kernel)

    convoled = jnp.convolve(inputs, kernel, mode = "same")

    print("Convolved 01:", convoled)

    convoled = jnp.convolve(inputs, kernel)

    print("Convolved 02:", convoled)

    convoled = jnp.convolve(inputs, kernel, mode = "valid")

    print("Convolved 03:", convoled)

    kernel = jnp.array([.1, .2, .3])

    convoled = jnp.convolve(inputs, kernel)

    print("Convolved 04:", convoled)

    convoled = jnp.convolve(inputs, kernel, mode = "full")

    print("Convolved 05:", convoled)

    convoled = jnp.convolve(inputs, kernel, mode = "same")

    print("Convolved 06: ", convoled)

    convoled = jnp.convolve(inputs, kernel, mode = "valid")

    print("Convolved 07:", convoled)

def main():

    convolve()

if __name__ == "__main__":

    main()
