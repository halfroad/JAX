import jax
import time
import jax.numpy as jnp

def aggregate(inputs):

    return jnp.sum(1.0 / (1 + jnp.exp(-inputs)))


number = 102400000.

begin = time.time()

x = jnp.arange(number)

print("x = ", x)

derivative_function = jax.grad(aggregate)
x_derivative = derivative_function(x)

print("x_derivative = ", x_derivative)

end = time.time()

print("Time {: .2f}s is consumed when iterating x {} times.".format(end - begin, len(x)))

begin = time.time()

x = jnp.arange(number)

derivative_function = jax.grad(aggregate)
derivative_function = jax.vmap(derivative_function)

x_derivative = derivative_function(x)

print("x_derivative = ", x_derivative)

end = time.time()

print("Time {: .2f}s is consumed when iterating x {} times.".format(end - begin, len(x)))

begin = time.time()

x = jnp.arange(number)

derivative_function = jax.grad(aggregate)
derivative_function = jax.vmap(derivative_function)
derivative_function = jax.jit(derivative_function)

x_derivative = derivative_function(x)

print("x_derivative = ", x_derivative)

end = time.time()

print("Time {: .2f}s is consumed when iterating x {} times.".format(end - begin, len(x)))
