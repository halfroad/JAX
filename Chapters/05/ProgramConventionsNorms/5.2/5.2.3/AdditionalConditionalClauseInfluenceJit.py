import jax


def function(length, val):

    return jax.numpy.ones((length, )) * val

def start():

    result = function(5, 4)

    print("result = ", result)

    """
    
    jit_function = jax.jit(function)
    jit_function_result = jit_function(5, 4)

    print("jit_function_result = ", jit_function_result)
    
    """


    jit_function = jax.jit(function, static_argnums = (0,))

    jit_function_result = jit_function(5, 4)

    print("jit_function_result = ", jit_function_result)



if __name__ == '__main__':

    start()
