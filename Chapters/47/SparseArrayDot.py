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
    
    dot(sparsedArray)
    
def dot(sparsedArray):

    array = jax.numpy.array([
        [1.],
        [2.],
        [3.],
    ])
    
    print("sparsedArray.T = ", sparsedArray)
    print("-----------------------------------------")
    
    dotted = sparsedArray.T @ array
    print("dotted = ", dotted)
    print("-----------------------------------------")
    
    # Normal matrixs required when computing by jax.numpy
    dotted = jax.numpy.dot(sparsedArray.T.todense(), array)
    print("dotted = ", dotted)
    print("-----------------------------------------")
    
def main():

    sparse()
    
if __name__ == "__main__":

    main()
