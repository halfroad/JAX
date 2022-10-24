from functools import partial

import jax.random
import tensorflow_datasets
from jax.example_libraries import stax, optimizers
from jax.experimental.ode import odeint


def sample_latent(key, shape):

    return jax.random.normal(key, shape = shape)

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
        stax.Relu,

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
def loss_generator(generator_parameters, discriminator_parameters, fake_image):

    # odeint(partial(loss_generator, generator_predict, discriminator_predict))

    generator_result = _generator_predict(generator_parameters, fake_image)
    fake_result = _discriminator_predict(discriminator_parameters, generator_result)

    # [0, 1] stands for fake
    fake_targets = jax.numpy.tile(jax.numpy.array([0, 1]), [fake_image.shape[0], 1])

    loss = jax.numpy.mean(jax.numpy.sum(-fake_targets * fake_result, axis = 1))

    return loss

@jax.jit
def loss_discriminator(generator_parameters, disciminator_parameters, fake_image, real_image):

    # odeint(partial(loss_discriminator, generator_predict, discriminator_predict))

    generator_result = _generator_predict(generator_parameters, fake_image)
    fake_result = _discriminator_predict(disciminator_parameters, generator_result)
    real_result = _discriminator_predict(disciminator_parameters, real_image)

    # Construct an array by repeating array the number of times given by reps.
    # [0, 1] means the fake
    fake_targets = jax.numpy.tile(jax.numpy.array([0, 1]), [fake_image.shape[0], 1])
    # [1, 0] means the real
    real_targets = jax.numpy.tile(jax.numpy.array([1, 0]), [real_image.shape[0], 1])

    loss = jax.numpy.mean(jax.numpy.sum(-fake_targets * fake_result, axis = 1)) + jax.numpy.mean(jax.numpy.sum(-real_targets * real_result, axis = 1))

    return loss

_generator_predict = None
_discriminator_predict = None

def train():

    (images_train, labels_train), (images_test, labels_test), batch_size, key, latent, generator_get_parameters, generator_predict, generator_state, generator_update_function, discriminator_get_parameters, discriminator_predict, discriminator_state, discriminator_update_function = setup()

    global _generator_predict
    _generator_predict = generator_predict
    global _discriminator_predict
    _discriminator_predict = discriminator_predict

    batch_number = len(images_train) // batch_size

    for i in range(1):

        for j in range(batch_number):

            begin = batch_size * j
            end = batch_size * (j + 1)

            real_image = images_train[begin: end]

            generator_parameters = generator_get_parameters(generator_state)
            discriminator_parameters = discriminator_get_parameters(discriminator_state)

            fake_image = jax.random.normal(key + j, shape = [batch_size, 1, 1, 1])

            grad_loss_generator = jax.grad(loss_generator)
            generator_gradient = grad_loss_generator(generator_parameters, discriminator_parameters, fake_image)

            generator_state = generator_update_function(j, generator_gradient, generator_state)

            grad_loss_discriminator = jax.grad(loss_discriminator)
            discriminator_gradient = grad_loss_discriminator(generator_parameters, discriminator_parameters, fake_image, real_image)

            discriminator_state = discriminator_update_function(j, discriminator_gradient, discriminator_state)

            print(f"Batch Number {j}")

def setup():

    dataset = tensorflow_datasets.load(name = "mnist", data_dir = "../../../Exclusion/Datasets/MNIST/", split = [tensorflow_datasets.Split.TRAIN, tensorflow_datasets.Split.TEST], batch_size = -1, as_supervised = True)

    (images_train, labels_train), (images_test, labels_test) = tensorflow_datasets.as_numpy(dataset)

    key = jax.random.PRNGKey(17)
    latent = sample_latent(key, shape = (100, 64))

    batch_size = 128

    real_shape = (-1, 28, 28, 1)
    fake_shape = (-1, 1, 1, 1)

    # Generator
    generator_init_random_parameters, generator_predict = generator()
    generator_init_function, generator_update_function, generator_get_parameters = optimizers.adam(step_size = 2e-4)
    _, generator_init_parameters = generator_init_random_parameters(key, fake_shape)
    generator_state = generator_init_function(generator_init_parameters)

    # Discriminator
    discriminator_init_random_parameters, discriminator_predict = discriminator()
    discriminator_init_function, discriminator_update_function, discriminator_get_parameters = optimizers.adam(step_size = 2e-4)
    _, discriminator_init_parameters = discriminator_init_random_parameters(key, real_shape)
    discriminator_state = discriminator_init_function(discriminator_init_parameters)

    images_train = (images_train - 256) / 256.

    return (images_train, labels_train), (images_test, labels_test), batch_size, key, latent, generator_get_parameters, generator_predict, generator_state, generator_update_function, discriminator_get_parameters, discriminator_predict, discriminator_state, discriminator_update_function

if __name__ == "__main__":

    train()
