import jax

class Counter:

    def __init__(self):

        self.n = 0

    def count(self):

        self.n += 1

        return self.n

    def reset(self):

        self.n = 0

class CounterV2:

    def __init__(self):

        pass

    def count(self, n):

        n += 1

        return n

def start():

    counter = Counter()

    fast_count = jax.jit(counter.count)

    print(jax.make_jaxpr(counter.count)())

    for _ in range(3):

        print(fast_count())

    counter = CounterV2()

    n = 0

    for i in range(3):

        n = counter.count(n)
        print(jax.make_jaxpr(counter.count)(n))

        print(n)

def main():

    start()

if __name__ == "__main__":

    main()
