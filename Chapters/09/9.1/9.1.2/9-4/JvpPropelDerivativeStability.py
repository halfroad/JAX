import jax
from jax import custom_jvp


def log_exp(x):

    return jax.numpy.log(1. + jax.numpy.exp(x))

@custom_jvp
def log_exp_v2(x, y):

    return jax.numpy.log(1. + jax.numpy.exp(x))

@log_exp_v2.defjvp
def log_exp_jvp(primals, tangents):

    x, y = primals
    x_dot, y_dot = tangents     # Custom derivative

    ans = log_exp_v2(x, y)
    ans_dot = (1. - 1. / (1. + jax.numpy.exp(x))) * x_dot

    return ans, ans_dot

def start():

    jit_log_exp = jax.jit(log_exp)
    print(jit_log_exp(3.))

    grad_log_exp = jax.grad(log_exp)
    jit_grad_log_exp = jax.jit(grad_log_exp)
    print(jit_grad_log_exp(3.))

    vmap_jit_grad_log_exp = jax.vmap(jit_grad_log_exp)
    array = jax.numpy.arange(4.)

    print(array)
    print(vmap_jit_grad_log_exp(array))
    print(grad_log_exp(99.))    # nan
    print(jax.numpy.exp(100.))  # inf

    ans, ans_dot = jax.jvp(log_exp_v2, (99., 1.), (99., 1.))

    print(ans, ans_dot)


def main():

    start()

if __name__ == "__main__":

    main()
