def FanInSum():
    
    init_fun = lambda prng, input_shape: (input_shape[0], ())
    apply_fun = lambda  params, inputs, **kwargs: sum(inputs)
    
    return init_fun, apply_fun

def model():
    
    fanInsum = FanInSum()