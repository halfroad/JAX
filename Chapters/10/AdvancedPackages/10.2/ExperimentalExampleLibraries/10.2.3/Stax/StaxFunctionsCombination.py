from jax.example_libraries import stax


def combine():

    init_random_params, predict = stax.serial(
        stax.Dense(1024),
        stax.Relu,
        stax.Dense(1024),
        stax.Relu,
        stax.Dense(10),
        stax.LogSoftmax
    )

    return init_random_params, predict

def train():

    init_random_params, predict = combine()

    print(init_random_params, predict)

if __name__ == '__main__':

    train()
