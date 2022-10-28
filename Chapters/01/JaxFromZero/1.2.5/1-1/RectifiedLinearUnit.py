# Rectified Linear Unit
import jax.numpy


def selu(x, alpha = 1.67, theta = 1.05):

    return jax.numpy.where(x > 0, x, (jax.numpy.exp(x) - 1) * alpha) * theta

if __name__ == "__main__":

    key = jax.random.PRNGKey(17)
    x_ = jax.random.normal(key, (5, ))

    print(selu(x_))
