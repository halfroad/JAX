import jax.lax


def cond_fun(x):

    print(x)

    return x < 17

def body_fun(x):

    print(x)

    return x + 1

def start():

    init_val = 0

    y = jax.lax.while_loop(cond_fun, body_fun, init_val)

    print(y)

if __name__ == '__main__':

    start()
