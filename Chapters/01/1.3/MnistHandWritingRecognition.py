import tensorflow as tf
import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp

from jax import jit, grad, random
from jax.example_libraries import optimizers, stax


def prepare():
    
    # tfds.core.utils.gcs_utils._is_gcs_disabled = True
    # tfds.display_progress_bar(enable = True)
    
    dataset = tfds.load("mnist", data_dir = "../Exclusion/Datasets/MNIST/", split = "train")
    
    images_train = dataset["train"]
    labels_train = dataset["test"]
    
    print("images_train.shape = {}, labels_train.shape = {}".format(images_train.shape, labels_train.shape))


def one_hot_no_jit(x, k = 10, dtype = jnp.float32):

    """

    Create a one-hot encoding of x of k.

    """

    return jnp.array(x[:, None] == jnp.arange(k), dtype)

# {Dense(1024) -> ReLU} x 2 -> Dense(10) -> Logsoftmax

def create_model():
    
    randomParameters, predictions = stax.serial()
    
def start():
    
    prepare()
    
start()
