import jax


"""

Paragraph 2.3.2, Auto Grad on JAX
Page 38


"""

def function(x):

    return x * x * x


def start():

    x = 1.0

    print(f"function = {function(x)}")
    print("-----------------------------------")

    grad_function = jax.grad(function)
    print(f"grad_function = {grad_function(x)}")
    print("-----------------------------------")

    grad_grad_function = jax.grad(grad_function)
    print(f"grad_grad_function = {grad_grad_function(x)}")
    print("-----------------------------------")

    grad_grad_grad_function = jax.grad(grad_grad_function)
    print(f"grad_grad_grad_function = {grad_grad_grad_function(x)}")

    x = jax.numpy.linspace(1, 5, 5)

    print(x)

    print(jax.numpy.sum(function(x)))

    print("-----------------------------------")
    # f'(x) = 3x²
    grad_sequence_function = jax.grad(lambda x_: jax.numpy.sum(function(x_)))

    print(grad_sequence_function(x))
    print("-----------------------------------")

if __name__ == "__main__":

    start()
