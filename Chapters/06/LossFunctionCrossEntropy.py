import jax

def cross_entropy(predictions, genuineness):

    # p(xᵢ) * log(q(xᵢ))
    # 1e-7: delta, tiny value to avoid jax.numpy.log(0) which incurs negative infinity
    entropys = genuineness * jax.numpy.log(predictions + 1e-7)
    
    # Σ
    entropys = -jax.numpy.sum(entropys, axis = -1)
    entropys = round(entropys, 3)
    
    return entropys
    
def test():

    predictions = jax.numpy.array([0.5, 0.2, 0.1, 0.9, 0.8, 0.1, 0.2, 0.3, 0.6, 0.7])
    genuineness = jax.numpy.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1])
    
    entropys = cross_entropy(predictions, genuineness)
    
    print(entropys)
    
if __name__ == "__main__":

    test()
