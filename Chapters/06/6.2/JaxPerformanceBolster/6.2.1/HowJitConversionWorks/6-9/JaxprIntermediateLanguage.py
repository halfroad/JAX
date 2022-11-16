import jax

global_list = []

def log(x):

    # This sentence breaks the rule of pure function, please be noted that this sentence was not executed
    global_list.append(x)

    ln_x = jax.numpy.log(x)
    ln_2 = jax.numpy.log(2.)

    return ln_x / ln_2

def main():

    jaxpr_log = jax.make_jaxpr(log)
    result = jaxpr_log(3.0)

    print(result)
    print(global_list)

if __name__ == '__main__':

    main()
