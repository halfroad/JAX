import jax.example_libraries.stax
from jax.example_libraries import stax, optimizers


def generator(features = 32):

    return jax.example_libraries.stax.serial(

        # Default input shape is [-1, 1, 1, 1]

        stax.ConvTranspose(features * 2, [3, 3], [2, 2]),
        stax.BatchNorm(),
        stax.Relu,

        stax.ConvTranspose(features * 4, [4, 4], [1, 1]),
        stax.BatchNorm(),
        stax.Relu,

        stax.ConvTranspose(features * 2, [3, 3], [2, 2]),
        stax.BatchNorm(),
        stax.Relu,

        stax.ConvTranspose(1, [4, 4], [2, 2]),
        stax.Tanh

        # Dimension generated is [-1, 28, 28, 1]
    )

def setup():

    key = jax.random.PRNGKey(15)

    return key

def generate(key, fake_image):

    init_random_params, predict = generator()

    fake_shape = (-1, 1, 1, 1)

    optimizer_init, optimizer_update, get_params = optimizers.adam(step_size = 2e-4)

    _, init_params = init_random_params(key, fake_shape)
    optimizer_state = optimizer_init(init_params)
    params = get_params(optimizer_state)

    predictions = predict(params, fake_image)

    return predictions

def main():

    key = setup()

    # Test for fake_image
    fake_image = jax.random.normal(key, shape = [10, 1, 1, 1])

    predictions = generate(key, fake_image)

    print(predictions.shape)

if __name__ == '__main__':

    main()
