import jax.numpy

"""

Paragraph 2.2.5, Activation Functions, softmax and Cross-Entropy
Page 30
Second Step: Implementation of softmax

"""

def softmax(inputs, axis = -1):

    unnormalized = jax.numpy.exp(inputs)

    # [20.085537    2.7182817   0.04978707]
    print(unnormalized)

    return unnormalized / unnormalized.sum(axis, keepdims = True)

def start():

    array = jax.numpy.array([3, 1, -3])
    normalized = softmax(array)

    # [0.8788782  0.11894322 0.00217852]
    print(normalized)

    normalized = jax.nn.softmax(array)

    # [0.87887824 0.11894324 0.00217852]
    print(normalized)

if __name__ == "__main__":

    start()
