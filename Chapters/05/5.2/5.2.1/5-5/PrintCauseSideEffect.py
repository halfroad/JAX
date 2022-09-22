import jax
import jax.numpy as jnp

def impure_print_side_effect(x):

    # Print is the side-effect
    print("Conduct function to compute")

    return x

def start():

    jit_impure_print_side_effect1 = jax.jit(impure_print_side_effect)

    print("First call: ", jit_impure_print_side_effect1(4.))
    print("----------------------------------")

    jit_impure_print_side_effect2 = jax.jit(impure_print_side_effect)

    print("Second call:", jit_impure_print_side_effect2(5.))
    print("----------------------------------")

    jit_impure_print_side_effect3 = jax.jit(impure_print_side_effect)
    print("Third call: ", jit_impure_print_side_effect3(jnp.array([5.])))

    """
    
    Conduct function to compute
    First call:  4.0
    ----------------------------------
    Second call: 5.0
    ----------------------------------
    Conduct function to compute
    Third call:  [5.]
    
    """

def main():

    start()


if __name__ == "__main__":

    main()
