import jax
from sklearn import datasets

"""

Paragraph 2.2.1, Iris Dataset Preparation
Page 23

"""

def setup():

    iris = datasets.load_iris()

    data = jax.numpy.float32(iris.data)
    targets = jax.numpy.float32(iris.target)

    return data, targets

def main():

    data, targets = setup()

    # print(f"Data: {data[: 15]}")
    print("----------------------")
    print(f"Target: {targets}")

if __name__ == "__main__":

    main()


