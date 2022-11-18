import jax


class Counter:

    def __init__(self):

        self.number = 0

    def count(self):

        self.number += 1

        return self.number

    def reset(self):

        self.number = 0

class CounterV2:

    def __init__(self):

        pass

    def count(self, number):

        number += 1

        return number

def start():

    counter = Counter()

    """

    for _ in range(3):

        print(counter.count())

    """

    jit_count = jax.jit(counter.count)

    for _ in range(3):

        print(jit_count())

    print("--------------------------")

    counter = CounterV2()

    number = 0

    for i in range(3):

        jit_count = jax.jit(counter.count)
        number = jit_count(number)

        print(number)

if __name__ == '__main__':

    start()
