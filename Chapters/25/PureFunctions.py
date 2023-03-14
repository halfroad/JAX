import jax

# Side Effect function
def impure_print_side_effect(inputs):
    
    # Side Effect
    print("Calculation")
    
    return inputs

g = 0.

def impure_saves_global(inputs):
    
    global g
    g = inputs
    
    return inputs

def pure_function_use_internal(inputs):
    
    state = dict(even = 0, odd = 0)
    
    for i in range(10):
        
        state["even" if i % 2 == 0 else "odd"] += inputs
        
    return state["even"] + state["odd"]

def test():
    
    '''

    impure_print_side_effect_jit = jax.jit(impure_print_side_effect)
    
    print("First call: ", impure_print_side_effect_jit(4.))
    print("-------------------------------------")
    
    print("Second call: ", impure_print_side_effect_jit(5.))
    print("-------------------------------------")
    
    print("Third call: ", impure_print_side_effect_jit(jax.numpy.array([5.])))
    print("-------------------------------------")
    
    print("Fourth call: ", impure_print_side_effect_jit(jax.numpy.array([6., 7.])))
    print("-------------------------------------")
    
    print("Fifth call: ", impure_print_side_effect_jit(jax.numpy.array([7., 8.])))
    
    impure_saves_global_jit = jax.jit(impure_saves_global)
    
    print("First call: ", impure_saves_global_jit(4.))
    print("Saved global g: ", g)
    
    '''
    
    pure_function_use_internal_jit = jax.jit(pure_function_use_internal)
    
    print(pure_function_use_internal_jit(3.))
    print(pure_function_use_internal_jit(jax.numpy.array([5.])))

if __name__ == "__main__":
    
    test()
