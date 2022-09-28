import jax

def batch_pooling(features, size = 2, stride = 2):

    # Number of dimensions
    assert features.ndim == 4, print("The input features should be 4 dimensions.")

    # Dimensions transform via einsum. (batch size, height, width, channels) to (batch size, channels, height, width)
    features = jax.numpy.einsum("bhwc->bchw", features)

    # Single image pooling
    def pooling(_features, _size = size, _stride = stride):

        channels = _features.shape[0]
        height = _features.shape[1]
        width = _features.shape[2]

        padding_height = round((height - _size + 1) / _stride)
        padding_width = round((width - _size + 1) / _stride)

        pool_out = jax.numpy.zeros((channels, padding_height, padding_width))

        for channel in range(channels):

            out_height = 0

            for i in jax.numpy.arange(0, height, _stride):

                out_width = 0

                for j in jax.numpy.arange(0, width, _stride):

                    pool_out.at[channel, out_height, out_width].set(jax.numpy.max(_features[channel, i: i + _size, j: j + _size]))
                    out_width = out_width + 1

                out_height = out_height + 1

        return pool_out

    vmap_pooling = jax.vmap(pooling)

    batch_pooling_output = vmap_pooling(features)
    batch_pooling_output = jax.numpy.einsum("bchw->bhwc", batch_pooling_output)

    return batch_pooling_output

def main():

    key = jax.random.PRNGKey(17)
    image = jax.random.normal(key, (10, 3, 50, 50))

    pooling = batch_pooling(image)

    print(pooling.shape)

if __name__ == "__main__":

    main()
