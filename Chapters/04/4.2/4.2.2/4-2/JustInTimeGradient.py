import jax
import time
import jax.numpy as jnp

@jax.jit

def aggregate(inputs):

    return jnp.sum(1.0 / (1.0 + jnp.exp(-inputs)))


begin = time.time()
x = jnp.arange(1024.)

print(x)

derivative_function = jax.grad(aggregate)
x_derivative = derivative_function(x)

print(x_derivative)

end = time.time()

print("Time {:.2f}s is consumed when iterating x {} times.".format(end - begin, len(x)))


def aggregate(inputs):

    return jnp.sum(1.0 / (1.0 + jnp.exp(-inputs)))


begin = time.time()

jit_aggregate = jax.jit(aggregate)

x = jnp.arange(1024.)

derivative_function = jax.grad(aggregate)

x_derivative = derivative_function(x)

print(x_derivative)

end = time.time()

print("Time {:.2f}s is consumed when iterating x {} times.".format(end - begin, len(x)))


def aggregate(inputs):

    return jnp.sum(1.0 / (1.0 + jnp.exp(-inputs)))


begin = time.time()

derivative_function = jax.grad(aggregate)
jit_derivative_function = jax.jit(derivative_function)

x = jnp.arange(1024.)

x_derivative = jit_derivative_function(x)

print(x_derivative)

end = time.time()

print("Time {:.2f}s is consumed when iterating x {} times.".format(end - begin, len(x)))
