import jax
from jax.example_libraries import optimizers

from DatasetBuilder import prepare
from Generator import generator
from Discriminator import discriminator
# from Loss import loss_generator, loss_discriminator

def sample_latent(key, shape):

    return jax.random.normal(key = key, shape = shape)

def setup():

    key = jax.random.PRNGKey(15)
    latent = sample_latent(key, shape = (100, 64))
    real_shape = (-1, 28, 28, 1)

    # Handle the generator_function
    generator_init_random_params, generator_predict = generator()

    fake_shape = (-1, 1, 1, 1)

    generator_optimizer_init, generator_optimizer_update, generator_get_params = optimizers.adam(step_size = 2e-4)

    _, generator_init_params = generator_init_random_params(key, fake_shape)
    generator_optimizer_state = generator_optimizer_init(generator_init_params)

    # Handle for the discriminator_function
    discriminator_init_random_params, discriminator_predict = discriminator()

    discriminator_optimizer_init, discriminator_optimizer_update, discriminator_get_params = optimizers.adam(step_size = 2e-4)

    _, discriminator_init_params = discriminator_init_random_params(key, real_shape)
    discriminator_optimizer_state = discriminator_optimizer_init(discriminator_init_params)

    batch_size = 128

    return (key, batch_size), (generator_predict, discriminator_predict), (generator_optimizer_update, discriminator_optimizer_update), (generator_get_params, discriminator_get_params), (generator_optimizer_state, discriminator_optimizer_state)

_generator_predict = None
_discriminator_predict = None

@jax.jit
def loss_generator(generator_params, discriminator_params, fake_images):

    generator_result = _generator_predict(generator_params, fake_images)
    fake_result = _discriminator_predict(discriminator_params, generator_result)
    fake_targets = jax.numpy.tile(jax.numpy.array([0, 1]), [fake_images.shape[0], 1])

    # [0,1 ] stands for fake_image
    losses = jax.numpy.sum(-fake_targets * fake_result, axis = 1)
    losses = jax.numpy.mean(losses)

    return losses

@jax.jit
def loss_discriminator(discriminator_params, generator_params, fake_images, real_images):

    generator_result = _generator_predict(generator_params, fake_images)
    fake_result = _discriminator_predict(discriminator_params, generator_result)
    real_result = _discriminator_predict(discriminator_params, real_images)

    # [0, 1] stands for fake image
    fake_targets = jax.numpy.tile(jax.numpy.array([0, 1]), [fake_images.shape[0], 1])
    # [1, 0] stands for real image
    real_targets = jax.numpy.tile(jax.numpy.array([1, 0]), [real_images.shape[0], 1])

    losses = jax.numpy.mean(jax.numpy.sum(-fake_targets * fake_result, axis = 1)) + jax.numpy.mean(jax.numpy.sum(-real_targets * real_result, axis = 1))

    return losses

def train():

    images, labels = prepare()
    (key, batch_size), (generator_predict, discriminator_predict), (generator_optimizer_update, discriminator_optimizer_update), (generator_get_params, discriminator_get_params), (generator_optimizer_state, discriminator_optimizer_state) = setup()

    global _generator_predict
    _generator_predict = generator_predict
    global _discriminator_predict
    _discriminator_predict = discriminator_predict

    batches_number = len(images) // batch_size

    for i in range(batches_number):

        start = batch_size * i
        end = batch_size * (i + 1)

        real_images = images[start: end]

        generator_params = generator_get_params(generator_optimizer_state)
        discriminator_params = discriminator_get_params(discriminator_optimizer_state)

        fake_images = jax.random.normal(key + i, shape = [batch_size, 1, 1, 1])

        grad_loss_generator = jax.grad(loss_generator)
        # def loss_generator(generator_params, discriminator_params, fake_images):
        generator_gradients = grad_loss_generator(generator_params, discriminator_params, fake_images)

        generator_optimizer_state = generator_optimizer_update(i, generator_gradients, generator_optimizer_state)

        grad_loss_discriminator = jax.grad(loss_discriminator)
        # def loss_discriminator(discriminator_params, generator_params, fake_images, real_images):
        discriminator_gradients = grad_loss_discriminator(discriminator_params, generator_params, fake_images, real_images)

        discriminator_optimizer_state = generator_optimizer_update(i, discriminator_gradients, discriminator_optimizer_state)

        if (i + 1) % 10 == 0:

            generator_losses = loss_generator(generator_params, discriminator_params, fake_images)
            discriminator_losses = loss_discriminator(discriminator_params, generator_params, fake_images, real_images)

            print(f"Batch number {i + 1} is completed")

if __name__ == '__main__':

    train()
