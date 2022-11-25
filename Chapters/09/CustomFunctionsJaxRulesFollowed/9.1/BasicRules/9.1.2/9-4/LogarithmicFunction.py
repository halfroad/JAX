import jax


def log_exp(x):

    exp = jax.numpy.exp(x)
    log = jax.numpy.log(1. + exp)

    return log

def run():

    jit_log_exp = jax.jit(log_exp)
    print("jit_log_exp(3.) = ", jit_log_exp(3.))

    grad_log_exp = jax.grad(log_exp)
    jit_grad_log_exp = jax.jit(grad_log_exp)
    print("jit_grad_log_exp(3.) = ", jit_grad_log_exp(3.))

    array = jax.numpy.arange(4.)
    vmap_jit_grad_log_exp = jax.vmap(jit_grad_log_exp)
    print("vmap_jit_grad_log_exp(3.) = ", vmap_jit_grad_log_exp(array))

    print("grad_log_exp(99.) = ", grad_log_exp(99.))

if __name__ == '__main__':

    run()
