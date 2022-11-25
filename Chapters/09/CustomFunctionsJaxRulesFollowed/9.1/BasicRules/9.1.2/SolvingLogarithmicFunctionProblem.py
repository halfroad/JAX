import jax


@jax.custom_jvp
def log_exp(x):

    exp = jax.numpy.exp(x)
    log = jax.numpy.log(1 + exp)

    return log

@log_exp.defjvp
def log_exp_jvp(primals, tangents):

    x, = primals
    x_dot, = tangents

    log = log_exp(x)
    log_dot = (1 - 1 / (1 + jax.numpy.exp(x))) * x_dot

    return log, log_dot

def run():

    grad_log_exp = jax.grad(log_exp)
    print("grad_log_exp(99.) = ", grad_log_exp(99.))

if __name__ == '__main__':

    run()
