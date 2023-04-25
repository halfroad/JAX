def FanOut(number):
    
    init_fun = lambda prng, input_shape:([input_shape] * number, ())
    apply_fun = lambda params, inputs, **kwargs: [inputs] * number
    
    return init_fun, apply_fun

def model():
    
    FanOut(number = 2)