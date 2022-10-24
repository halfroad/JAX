import jax
from jax.example_libraries import stax, optimizers


def generator(features = 32):

    return stax.serial(

        # Default input dimension is [-1, 1, 1, 1]
        stax.ConvTranspose(features * 2, [3, 3], [2, 2]),
        stax.BatchNorm(),
        stax.Relu,

        stax.ConvTranspose(features * 4, [4, 4], [1, 1]),
        stax.BatchNorm(),
        stax.Relu,

        stax.ConvTranspose(features * 2, [3, 3], [2, 2]),
        stax.BatchNorm(),

        stax.ConvTranspose(1, [4, 4], [2, 2]),
        stax.Tanh

        # Dimension generated is [-1, 28, 28, 1]
    )

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

@jax.jit
def loss_generator(generator_parameters, generator_predict, discriminator_parameters, discriminator_predict, fake_image):

    generator_result = generator_predict(generator_parameters, fake_image)
    fake_result = discriminator_predict(discriminator_parameters, generator_result)

    fake_targets = jax.numpy.tile(jax.numpy.array([0, 1]), [fake_image.shape[0], 1])

    # [1, 1] stands for fake
    loss = jax.numpy.mean(jax.numpy.sum(-fake_targets * fake_result, axis = 1))

    return loss

@jax.jit
def loss_discriminator(disciminator_parameters, generator_predict, disciminator_predict, generator_parameters, fake_image, real_image):

    generator_result = generator_predict(generator_parameters, fake_image)
    fake_result = disciminator_predict(disciminator_parameters, generator_result)
    real_result = disciminator_predict(disciminator_parameters, real_image)

    # Construct an array by repeating array the number of times given by reps.
    # [0, 1] means the fake
    fake_targets = jax.numpy.tile(jax.numpy.array([0, 1]), [fake_image.shape[0], 1])
    # [1, 0] means the real
    real_targets = jax.numpy.tile(jax.numpy.array([1, 0]), [real_image.shape[0], 1])

    loss = jax.numpy.mean(jax.numpy.sum(-fake_targets * fake_result, axis = 1)) + jax.numpy.mean(jax.numpy.sum(-real_targets * real_result), axis = 1)

def start():

    key = jax.random.PRNGKey(17)

    # Handle the fake_image
    fake_image = jax.random.normal(key, shape = [10, 1, 1, 1])

    init_random_parameters, predict = generator()

    fake_shape = (-1, 1, 1, 1)

    init_function, update_function, get_parameters = optimizers.adam(step_size = 2e-4)
    _, init_parameters = init_random_parameters(key, fake_shape)

    opt_state = init_function(init_parameters)
    parameters = get_parameters(opt_state)

    result = predict(parameters, fake_image)

    print(result.shape)

    real_image = jax.random.normal(key, shape = [10, 28, 28, 1])
    init_random_parameters, predict = discriminator()

    real_shape = [-1, 28, 28, 1]

    init_function, update_function, get_parameters = optimizers.adam(step_size = 2e-4)

    _, init_parameters = init_random_parameters(key, real_shape)

    opt_state = init_function(init_parameters)
    parameters = get_parameters(opt_state)

    result = predict(parameters, real_image)

    print(result.shape)

if __name__ == "__main__":

    start()
