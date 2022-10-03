import jax


@jax.custom_jvp
def function(x):

    return x

def fucntion_jvp(primals, tangents):

    x, = primals
    t, = tangents

    return function(x), t * x

def start():

    function.defjvp(fucntion_jvp)

    print(function(3.))

    grad_function = jax.grad(function)
    print(grad_function(2.))

    y, y_dot = jax.jvp(function, (3., ), (2., ))

    print(y, y_dot)


def main():

    start()

if __name__ == "__main__":

    main()


