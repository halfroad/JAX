import jax
import jax.numpy as jnp

global_list = []

# Impure function
def log(x):

    # This breaks the rule of pure function.
    global_list.append(x)

    ln_x = jnp.log(x)
    ln_2 = jnp.log(2.)

    return ln_x / ln_x

def print_log(x):

    print("Print test:", x)

    x = jnp.log(x)

    print("Print test", x)

    return x

def start():

    jaxpr_log = jax.make_jaxpr(log)

    result = jaxpr_log(3.)

    print(result)
    print(global_list)

    jaxpr_print_log = jax.make_jaxpr(print_log)

    result = jaxpr_print_log(3.)

    print(result)

def main():

    start()

if __name__ == "__main__":

    main()

