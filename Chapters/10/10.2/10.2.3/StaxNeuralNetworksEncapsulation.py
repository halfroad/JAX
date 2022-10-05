from jax.example_libraries import stax


def model():

    init_random_parameters, predict = stax.serial(

        stax.Dense(1024),
        stax.Relu,
        stax.Dense(1024),
        stax.Relu,
        stax.Dense(10),
        stax.LogSoftmax
    )

    return init_random_parameters, predict

def start():

    init_random_parameters, predict = model()

    print(init_random_parameters, predict)

def main():

    start()

if __name__ == "__main__":

    main()
