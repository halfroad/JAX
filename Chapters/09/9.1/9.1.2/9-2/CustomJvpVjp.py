import jax
from jax import custom_jvp


def function1(x, y):

    return x * y

# Explicitly attribute the method to be custom derivative needed methodlghhlphph
@custom_jvp
def function2(x, y):

    return x * y

# Attribute the function is custom result of derivatives and cauculation
@function2.defjvp
def function_jvp(primals, tangents):

    # x and y that inputs
    x, y = primals

    # x_dot and y_dot that inputs
    x_dot, y_dot = tangents

    # the result computed by forward function
    primal_out = function2(x, y)

    # Custom derivative function
    tangent_out = x_dot + y_dot

    # Return the computed function and custom derivative function, and derivate the parameters offered by primals
    return primal_out, tangent_out


def start():

    print(function1(2., 3.))

    grad_function = jax.grad(function1)

    print(grad_function(2., 3.))

    y, y_dot = jax.jvp(function2, (2., 3.), (5., 6.))

    print("y =", y, "y_dot =", y_dot)

    grad_function = jax.grad(function2, argnums = [0, 1])

    print("After the custom function_jvp of JVP =", grad_function(2., 3.))

    grad_function = jax.grad(function1, argnums = [0, 1])
    print("Original JAX derivative function =", grad_function(2., 3.))

def main():

    start()

if __name__ == "__main__":

    main()
