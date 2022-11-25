import jax

@jax.custom_jvp
def function(x, y):

    return x * y

def function1(x, y):

    return x * y

@function.defjvp
def function_jvp(primals, tangents):

    x, y = primals
    x_dot, y_dot = tangents

    primal_out = function(x, y)

    # Custom derivative function, intrinsic grad provided by JAX
    tangent_out = y_dot + x_dot

    return primal_out, tangent_out

def test():

    derivative = jax.jvp(function, (2., 3.), (2., 3.))
    print("jax.jvp derivative = ", derivative)

    grad_function = jax.grad(function, argnums = [0, 1])
    derivative = grad_function(2., 3.)
    print("Derivative by jax.grad = ", derivative)

    grad_function = jax.grad(function1, argnums = [0, 1])
    derivative = grad_function(2., 3.)
    print("Derivative by jax.grad = ", derivative)

if __name__ == '__main__':

    test()
