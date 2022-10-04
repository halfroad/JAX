import jax
from jax import custom_jvp


@custom_jvp
def function(x, y):

    return x * y

@function.defjvp
def function_jvp(primals, tangents):

    x, y = primals
    x_dot, y_dot = tangents

    primal_out = function(x, y)
    tangent_out = y * x_dot + x * y_dot

    return primal_out, tangent_out

@custom_jvp
def function2(x):

    return 2 * x

@custom_jvp
def function3(x):

    return 2 * x

@function3.defjvp
def function3_jvp(primals, tangents):

    x, = primals
    x_dot, = tangents

    if x >= 0:

        return function3(x), x_dot

    else:

        return function3(x), 2 * x_dot

def start():

    print(function(4., 5.))

    grad_function = jax.grad(function)

    print(grad_function(2., 3.))

    # function2 = lambda x: x * 2

    function2.defjvp(lambda primals, tangents: primals * 2)

    grad_function = jax.grad(function2)

    print(grad_function(3.))

    grad_function = jax.grad(function3)

    print(grad_function(1.))
    print(grad_function(-1.))

def main():

    start()

if __name__ == "__main__":

    main()
