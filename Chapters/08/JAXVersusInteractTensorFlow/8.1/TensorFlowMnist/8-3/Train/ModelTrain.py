import time

import tensorflow as tf

def train(mnist_model, trains, tests, accelerator):

    # Indicate that the GPU is used to compute the model while running TensorFlow
    with tf.device(accelerator):

        image = tf.keras.Input(shape = (28, 28, 1))
        logits = mnist_model(image)

        model = tf.keras.Model(image, logits)

        for i in range(4):

            start = time.time()

            model.compile(optimizer = tf.keras.optimizers.SGD(1e-3), loss = tf.keras.losses.categorical_crossentropy, metrics = ["accuracy"])
            model.fit(trains, epochs = 50, validation_data = tests, verbose = 0)

            end = time.time()

            loss, accuracy = model.evaluate(tests)

            print("Test Loss", loss)
            print("Accuracy", accuracy)

            print(f"Test {i + 1} started, %.12fs is consumed" % (end - start))


def test():

    train()

if __name__ == '__main__':

    test()
