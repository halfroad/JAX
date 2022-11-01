import jax.numpy

"""

Paragraph 2.2.5, Activation Functions, softmax and Cross-Entropy
Page 29
First Step: Activation Functions

"""

def activate(inputs):

    return jax.numpy.tanh(inputs)
