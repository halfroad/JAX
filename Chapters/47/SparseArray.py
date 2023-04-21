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
    
    denseArray = sparsedArray.todense()
    print("denseArray = ", denseArray)
    print("--------------------------------------")
    
    print("sparsedArray = ", sparsedArray)
    print("sparsedArray.data = ", sparsedArray.data)
    print("--------------------------------------")
    
    print("sparsedArray.indices = ", sparsedArray.indices)
    print("--------------------------------------")
        
    for tuple in sparsedArray.indices:
    
        print(f"array[{tuple[0]}, {tuple[1]}] = ", array[tuple[0], tuple[1]])
        
    print("sparsedArray.ndim = ", sparsedArray.ndim)
    print("sparsedArray.shape = ", sparsedArray.shape)
    print("sparsedArray.dtype = ", sparsedArray.dtype)
    print("sparsedArray.nse = ", sparsedArray.nse)
    
    print("-----------------------------------------")
    
    dot(sparsedArray)
    
def dot(sparsedArray):

    array = jax.numpy.array([
        [1.],
        [2.],
        [3.],
    ])
    
    print("sparsedArray.T = ", sparsedArray)
    print("-----------------------------------------")
    
    dotted = sparsedArray.T@array
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
