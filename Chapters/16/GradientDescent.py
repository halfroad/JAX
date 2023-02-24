import jax

def gradient_function(inputs, genuines, theta):
    
    predictions = jax.numpy.dot(inputs, theta) - genuines
    transposed = jax.numpy.transpose(inputs)
    rows = inputs.shape[0]
    
    gradient = 1. / rows * jax.numpy.dot(transposed, predictions)
    
    return gradient
 
def test():

    inputs = jax.numpy.array([
        [[0., 1., 2., 3.],
        [4., 5., 6., 7.],
        [8., 9. ,10., 11.]],
        
        [[12., 13., 14., 15.],
        [16., 17., 18., 19.],
        [20., 21., 22., 23.]],
        
        [[24., 25., 26., 27.],
        [28., 29., 30., 31.],
        [32., 33., 34., 35.]],
        
        [[36., 37., 38., 39.],
        [40., 41., 42., 43.],
        [44., 45., 46., 47.]],
        
        [[48., 49., 50., 51.],
        [52., 53., 54., 55.],
        [56., 57., 58., 59.]]
        ])
        
    print("Shape of inputs = ", inputs.shape)

    transposed = jax.numpy.transpose(inputs, ((1, 2, 0)))
    print("Shape of transposed = ", transposed.shape)
    print("Transposed = ", transposed)

    transposed = jax.numpy.transpose(inputs)
    print("Shape of transposed = ", transposed.shape)
    print("Transposed = ", transposed)

if __name__ == "__main__":

    test()
