import time
import tensorflow as tf

import sys
sys.path.append("../8-1/")
sys.path.append("../8-2/")
from TensorFlowMnistDatasetPreparation import setup
from MnistNeuralNetworksCompoents import MnistNeuralNetworksCompoentsLayers

def train(trains, tests):

    # Tell the TensorFlow to use GPU to train the model
    # with.tfdevice("/CPU:0")
    with tf.device("/GPU:0"):

        image = tf.keras.Input(shape = (28, 28, 1))
        logits = MnistNeuralNetworksCompoentsLayers()(image)
        model = tf.keras.Model(image, logits)

        print(model.summary())

        begin = time.time()

        model.compile(optimizer = tf.keras.optimizers.SGD(1e-3), loss = tf.keras.losses.categorical_crossentropy, metrics = ["accuracy"])

        for i in range(4):

            model.fit(trains, epochs = 50, validation_data = tests, verbose = 0)

            end = time.time()

            loss, accuracy = model.evaluate(tests)

            print(f"Iteration {i + 1} is completed, {end - begin}s is consumed, loss = {loss}, accuracy = {accuracy}")

def start():

    trains, tests = setup()

    train(trains, tests)

def main():

    start()

if __name__ == "__main__":

    main()
