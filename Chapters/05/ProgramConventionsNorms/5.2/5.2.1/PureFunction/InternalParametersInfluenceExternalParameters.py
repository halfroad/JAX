import jax

g = 0.

def impure_saves_global(x):

    global g

    g = x

    return x

def main():

    jit_impure_saves_gloabl = jax.jit(impure_saves_global)

    print("first call: ", jit_impure_saves_gloabl(4.))
    print("Saved global: ", g)

if __name__ == '__main__':

    main()
