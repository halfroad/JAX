import jax.lax

def body_func(i, x):

    print(i, x)

    return x + 1

def start():

    init_val = 0

    begin = 0
    stop = 10

    # Here, 2 parameters are passed
    body_fun = lambda i, x: x + i

    result = jax.lax.fori_loop(begin, stop, body_fun, init_val)

    print("result = ", result)

    result = jax.lax.fori_loop(begin, stop, body_func, init_val)

    print("result = ", result)

    jax.lax.scan

if __name__ == '__main__':

    start()
