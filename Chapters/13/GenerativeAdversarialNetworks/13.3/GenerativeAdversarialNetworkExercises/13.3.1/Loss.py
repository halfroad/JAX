import jax

@jax.jit
def loss_generator(generator_params, generator_predict, discriminator_params, discriminator_predict, fake_images):

    generator_result = generator_predict(generator_params, fake_images)
    fake_result = discriminator_predict(discriminator_params, generator_result)
    fake_targets = jax.numpy.tile(jax.numpy.array([0, 1]), [fake_images.shape[0], 1])

    # [0,1 ] stands for fake_image
    losses = jax.numpy.sum(-fake_targets * fake_result, axis = 1)
    losses = jax.numpy.mean(losses)

    return losses

@jax.jit
def loss_discriminator(discriminator_params, discriminator_predict, generator_params, generator_predict, fake_images, real_images):

    generator_result = generator_predict(generator_params, fake_images)
    fake_result = discriminator_predict(discriminator_params, generator_result)
    real_result = discriminator_predict(discriminator_params, real_images)

    # [0, 1] stands for fake image
    fake_targets = jax.numpy.tile(jax.numpy.array([0, 1]), [fake_images.shape[0], 1])
    # [1, 0] stands for real image
    real_targets = jax.numpy.tile(jax.numpy.array([1, 0]), [real_images.shape[0], 1])

    losses = jax.numpy.mean(jax.numpy.sum(-fake_targets * fake_result, axis = 1)) + jax.numpy.mean(jax.numpy.sum(-real_targets * real_result, axis = 1))

    return losses

