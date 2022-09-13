import jax.numpy as jnp
import jax

def cross_entropy(genuines, predicted):
    
    """

    Cross Entropy Function
    
    p(x) = Genuine Distribution Probabilities
    q(x) = Predicted Distribution Probabilities
    
    H(p, q) = - SUM[i = 1, n](p(Xi)log(q(Xi)))
    
    The difference between p(x) and q(x)
    
    genuines: genuine labels
    predicted: outputs of neural networks
    
    """
    
    genuines = jnp.array(genuines)
    predicted = jnp.array(predicted)
    
    print(genuines)
    print(jnp.log(predicted + 1e-7))
    
    print(genuines * jnp.log(predicted + 1e-7))
    
    difference = -jnp.sum(genuines * jnp.log(predicted + 1e-7), axis = -1)
    
    return round(difference, 3)
    
def main():
    
    predicted = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    genuines = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    
    difference = cross_entropy(genuines, predicted)
    
    print("difference = {}".format(difference))    
    
if __name__ == "__main__":
    
    main()
    
    