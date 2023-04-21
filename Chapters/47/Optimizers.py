import jax.example_libraries.optimizers

def model():

    optimizer_init_function, optimzier_update_function, get_params_function = jax.example_libraries.optimizers.adam(step_size = 1e-3)
    
    _, initial_params = init_random_params(key, input_shape)
    
    optimizer_state = optimizer_init_function = optimizer_init_function(initial_params)
    # ...
    
    params = get_params_function(optimizer_state)
    loss_function_grad = jax.grad(loss_function)
    gradients = loss_function_grad(params, inputs, targets)
    # Optimize the inputs and update the params
    optimizer_state = optimzier_update_function(_, gradients, optimizer_state)
    
def main():

    model()
    
if __name__ == "__main__":

    main()
