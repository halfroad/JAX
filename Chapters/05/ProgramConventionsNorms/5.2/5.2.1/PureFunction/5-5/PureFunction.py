import jax


def impure_print_side_effect(x):

    """

    Function that brings side effect

    """
    # This is a side effect
    print("Conduct the function compuation")

    return x

def main():

    jit_impure_print_side_effect = jax.jit(impure_print_side_effect)

    print("First call: ", jit_impure_print_side_effect(4.0))
    print("---------------------------------")

    jit_impure_print_side_effect = jax.jit(impure_print_side_effect)

    print("First call: ", jit_impure_print_side_effect(5.0))
    print("---------------------------------")

    jit_impure_print_side_effect = jax.jit(impure_print_side_effect)

    print("Third call: ", jit_impure_print_side_effect([5.0]))
    print("---------------------------------")

if __name__ == "__main__":

    main()
