import jax.numpy as jnp

"""

Mathmatical formular of Gradient Descent

"""

def chain(x, gama = 0.1):
    
    x = x - gama * 2 * x
    
    return x

def main():
    
    x = 1
    
    gradiences = []
    
    for _ in range(4):
        
        x = chain(x)
        
        gradiences.append(x)
        
        print(x)
        
        
if __name__ == "__main__":
    
    main()