import jax.numpy

"""

Paragraph 2.2.5, Activation Functions, softmax and Cross-Entropy
Page 31
Third Step: Implementation of Cross-Entropy

"""

def cross_entropy(genuines, predictions, delta = 1e-7):

    predictions = predictions + delta
    logs = jax.numpy.log(predictions)
    crosses = genuines * logs

    entropys = -jax.numpy.sum(crosses, axis = -1)

    return round(entropys, 3)

def start():

    genuines = jax.numpy.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    predictions = jax.numpy.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

    entropys = cross_entropy(genuines, predictions)

    print(entropys)

if __name__ == "__main__":

    start()
