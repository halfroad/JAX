import jax
from jax.experimental import sparse

def sparse():

    array = jax.numpy.array([
        [0., 1., 0., 2.],
        [3., 0., 0., 0],
        [0., 0., 4., 0.],
    ])
    
    print("array = ", array)
    sparsedArray = jax.experimental.sparse.BCOO.fromdense(array)
    print("--------------------------------------")
    
    return sparsedArray
 
@jax.jit
def function(array, sparsedArray):
    
    dotted = sparsedArray.T @ array
    dotted = dotted.sum()
    
    return dotted
    
def main():

    array = jax.numpy.array([
    
        [1.],
        [2.],
        [3.],
    ])
    
    sparsedArray = sparse()
    
    dotted = function(array, sparsedArray)
    
    print("dotted = ", dotted)
    print("---------------------------------")
    
    function_sparsify = jax.experimental.sparse.sparsify(function)
    dotted = function_sparsify(array, sparsedArray)
    
    print("dotted = ", dotted)
    print("---------------------------------")
    
    
if __name__ == "__main__":

    main()
