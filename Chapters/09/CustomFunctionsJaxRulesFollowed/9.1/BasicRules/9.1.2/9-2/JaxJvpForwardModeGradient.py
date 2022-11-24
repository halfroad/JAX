import jax


def function(x, y):

    return x * y

def test():

    print("function(2., 3.) = ", function(2., 3.))

    grad_function = jax.grad(function)

    derivative = grad_function(2., 3.)

    print("derivative = ", derivative)

if __name__ == '__main__':

    test()
