import jax


@jax.custom_jvp
def function(x, y):

    return x * y

@function.defjvp    # Indicate the custom derivative and computation results
def funciton_jvp(primals, tangents):

    x, y = primals                      # The input x and y
    x_dot, y_dot = tangents             # The input x_dot, y_dot

    primal_out = function(x, y)         # Compute the result of forward function
    tangent_out = y_dot + x_dot         # Custom derivative funciton

    # Return the computed function and custom derivative function, and derivate the parameters offered by primals
    return primal_out, tangent_out

def test():

    y, y_dot = jax.jvp(function, (2., 3.), (2., 3.))

    print(y, y_dot)

if __name__ == '__main__':

    test()
