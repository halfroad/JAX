import jax

def test():
    
    print(jax.numpy.add(1, 1.0))
    
    print(jax.lax.add(1, 1))
    # print(jax.lax.add(1, 1.0))
    
    print(jax.lax.add(jax.numpy.float32(1), 1.0))
    
if __name__ == "__main__":
    
    test()
