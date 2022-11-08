"""

Paragraph 3.2.3
Page 47
3rd Step: Mathematic Compution of Gradient Descending

"""


def chain(x, alpha = 0.1):

    x = x - alpha * 2 * x

    return x

def chain2(x1, x2, alpha = 0.1):

    result = x1 - alpha * 2 * x1 + x2 - alpha * 2 * x2

    return result

def start():

    x = 1

    for _ in range(10):

        x = chain(x)

        print(x)

    x1 = 2
    x2 = 5

    for _ in range(4):

        x1 = chain(x1, alpha = 0.3)
        x2 = chain(x2, alpha = 0.3)

        print(f"{x1, x2}")


if __name__ == "__main__":

    start()

