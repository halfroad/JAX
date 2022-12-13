import jax.random
from jax.example_libraries import stax, optimizers


def discriminator(features = 32):

    return stax.serial(

        stax.Conv(features, [4, 4], [2, 2]),
        stax.BatchNorm(),
        stax.LeakyRelu,

        stax.Conv(features, [4, 4], [2, 2]),
        stax.BatchNorm(),
        stax.LeakyRelu,

        stax.Conv(2, [4, 4], [2, 2]),
        stax.Flatten
    )

def setup():

    key = jax.random.PRNGKey(15)
    real_shape = (-1, 28, 28, 1)

    return key, real_shape

def discriminate(key, real_shape, real_image):

    init_random_params, predict = discriminator()
    optimizer_init, optimzier_update, get_params = optimizers.adam(step_size = 2e-4)

    _, init_params = init_random_params(key, real_shape)
    optmizer_state = optimizer_init(init_params)
    params = get_params(optmizer_state)

    predictions = predict(params, real_image)

    return predictions

def main():

    key, real_shape = setup()

    real_image = jax.random.normal(key, shape = [10, 28, 28, 1])
    predictions = discriminate(key, real_shape, real_image)

    print(predictions)

if __name__ == '__main__':

    main()
