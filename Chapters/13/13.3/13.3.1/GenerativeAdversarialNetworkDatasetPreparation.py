import functools

import datasets
import jax
import tensorflow_datasets


def setup():

    dataset = tensorflow_datasets.load(name ="mnist", data_dir = "../../../../Exclusion/Datasets/MNIST/", split = [tensorflow_datasets.Split.TRAIN, tensorflow_datasets.Split.TEST], batch_size = -1, as_supervised = True)
    (images_train, labels_train), (images_test, labels_test) = tensorflow_datasets.as_numpy(dataset)

    images_train = (images_train - 256.0) / 256.0

    return (images_train, labels_train), (images_test, labels_test)

def general_convolution_transpose(dimension_numbers, out_channels, filters_shape, strides = None, padding = "VALID", weights_init = None, bias_init = jax.random.normal(1e-6)):

    lhs_spec, rhs_spec, out_spec = dimension_numbers

    Convolution1DTranspose = functools.partial(general_convolution_transpose, ("NHC", "HIO", "NHC"))
    ConvolutionTranspose = functools.partial(general_convolution_transpose, ("NHWC", "HWIO", "NHWC"))

def start():

    (images_train, labels_train), (images_test, labels_test) = setup()

    print(images_train.shape, labels_train.shape, images_test.shape, labels_test.shape)

if __name__ == "__main__":

    start()
