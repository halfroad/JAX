def Identity():
    
    init_fun = lambda prng, inputt_shape: (input_shape, ())
    apply_fun = lambda params, inputs, **kwargs: inputs
    
    return init_fun, apply_fun