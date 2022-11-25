import jax


@jax.custom_jvp
def function(x, y):

    return x * y

@function.defjvp
def function_jvp(primals, tangents):

    x, y = primals
    x_dot, y_dot = tangents

    primal_out = function(x, y)
    tangent_out = y * x_dot + x * y_dot

    return primal_out, tangent_out

def run():

    grad_function = jax.grad(function)

    # Be noted here only the function is derivated
    print("grad_function(2., 3.) = ", grad_function(2., 3.))

if __name__ == '__main__':

    run()
