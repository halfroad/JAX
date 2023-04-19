import jax

def logxp(x):
    
    return jax.numpy.log(1. + jax.numpy.exp(x))

@jax.custom_jvp
def logxp1(x):
    
    return jax.numpy.log(1. + jax.numpy.exp(x))

@logxp1.defjvp
def logxp1_jvp(primals, tangents):
    
    x, = primals
    x_dot, = tangents
    
    result = logxp(x)
    result_dot = (1 - 1 / (1 + jax.numpy.exp(x))) * x_dot
    
    return result, result_dot

def test():
    
    logxp_jit = jax.jit(logxp)
    
    result = logxp_jit(3.)
    
    print("result = ", result)
    print("-------------------")
    
    logxp_grad = jax.grad(logxp)
    log_grad_jit = jax.jit(logxp_grad)
    
    result = log_grad_jit(3.)
    
    print("result = ", result)
    print("-------------------")
    
    log_grad_jit_vmap = jax.vmap(log_grad_jit)
    
    result = log_grad_jit_vmap(jax.numpy.arange(4.))
    
    print("result = ", result)
    print("-------------------")
    
    result = logxp_grad(99.)
    
    print("result = ", result)
    print("-------------------")
    
    logxp_grad = jax.grad(logxp1)
    result = logxp_grad(99.)
    
    print("result = ", result)
    print("-------------------")
    
def main():
    
    test()
    
if __name__ == "__main__":
    
    main()
    
    
    
    