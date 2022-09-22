import jax
import jax.numpy as jnp
import time

def exam_array_bound():

    array = jnp.arange(9)

    print(array)
    print(array[-1])
    print(array[11])

def add():

    begin = time.time()

    array = jnp.arange(10000)
    aggregation = jnp.sum(array)

    end = time.time()

    print("aggregation = {} with computing time is {}. ".format(aggregation, end - begin))

    # Error
    #array = range(9)
    #aggregation = jnp.sum(array)

    #print(aggregation)

    begin = time.time()

    array = jnp.arange(10000)
    array = jnp.array(array)
    aggregation = jnp.sum(array)

    end = time.time()

    print("aggregation = {} with computing time is {}. ".format(aggregation, end - begin))

def main():

    exam_array_bound()
    add()

if __name__ == "__main__":

    main()
