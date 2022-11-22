import jax


def batch_pooling(features, size = 2, stride = 2):

    assert features.ndim == 4, print("The input dimensions must be 4")
    
    features = jax.numpy.einsum("bhwc->bchw", features)

    # Below internal function implements the single pooling
    def pooling(features_ = features, size_ = size, stride_ = stride):

        shape = features_.shape

        channels = shape[0]
        height = shape[1]
        width = shape[2]

        padding_height = round((height - size_ + 1) / stride_)
        padding_width = round((width - size_ + 1) / stride_)

        pooling_output = jax.numpy.zeros((channels, padding_height, padding_width))

        for i in range(channels):

            out_height = 0

            for j in jax.numpy.arange(0, height, stride_):

                out_width = 0

                for k in jax.numpy.arange(0, width, stride):

                    pooling_output = pooling_output.at[i, out_height, out_width].set(jax.numpy.max(features_[i, j: j + size_, k: k + size_]))
                    out_width += 1

                out_height += 1

        return pooling_output

    vmap_batch_pooling = jax.vmap(batch_pooling)

    batch_pooling_output = vmap_batch_pooling(features)
    batch_pooling_output = jax.numpy.einsum("bchw->bhwc", batch_pooling_output)

    return batch_pooling_output

