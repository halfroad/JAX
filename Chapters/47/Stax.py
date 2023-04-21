import jax.example_libraries.stax

def model():

    init_random_params, predict = jax.example_libraries.stax.serial(
        
            jax.example_libraries.stax.Dense(1024),
            jax.example_libraries.stax.Relu,
            
            jax.example_libraries.stax.Dense(1024),
            jax.example_libraries.stax.Relu,
            
            jax.example_libraries.stax.Dense(10),
            
            jax.example_libraries.stax.LogSoftmax,
        )
    
def main():

    model()
    
if __name__ == "__main__":

    main()
