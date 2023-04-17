import tensorflow as tf
import numpy as np
import time

def setup():
    
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    
    print((train_images.shape, train_labels.shape), (test_images.shape, test_labels.shape))
    
    train_images = np.expand_dims(train_images, axis = 3)
    test_images = np.expand_dims(test_images, axis = 3)
    
    train_labels = tf.one_hot(train_labels, depth = 10)
    test_labels = tf.one_hot(test_labels, depth = 10)
    
    trains = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(1024).batch(256)
    tests = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).shuffle(1024).batch(256)
    
    return (trains, (train_images, train_labels)), (tests, (test_images, test_labels))

class MnistNeuralNetwork(tf.keras.layers.Layer):
    
    def __init__(self):
        
        super(MnistNeuralNetwork, self).__init__()
        
    def build(self, input_shape):
        
        self.conv1 = tf.keras.layers.Conv2D(filters = 1, kernel_size = 3, activation = tf.nn.relu)
        self.batchNormalization1 = tf.keras.layers.BatchNormalization()
        
        self.conv2 = tf.keras.layers.Conv2D(filters = 1, kernel_size = 3, activation = tf.nn.relu)
        self.batchNormalization2 = tf.keras.layers.BatchNormalization()
        
        self.dense = tf.keras.layers.Dense(10, activation = tf.nn.sigmoid)
        
        super(MnistNeuralNetwork, self).build(input_shape = input_shape)
        
    def call(self, inputs):
        
        embedding = inputs
        
        embedding = self.conv1(embedding)
        embedding = self.batchNormalization1(embedding)
        
        embedding = self.conv2(embedding)
        embedding = self.batchNormalization2(embedding)
        
        embedding = tf.keras.layers.Flatten()(embedding)
        
        logits = self.dense(embedding)
        
        return logits
    
def train():
    
    #with tf.device("/GPU:0"):
    with tf.device("/CPU:0"):
        
        image = tf.keras.Input(shape = (28, 28, 1))
        logits = MnistNeuralNetwork()(image)
        model = tf.keras.Model(image, logits)
        
        iterations = 4
        
        (trains, (train_images, train_labels)), (tests, (test_images, test_labels)) = setup()
        
        for i in range(iterations):
            
            start = time.time()
            
            model.compile(optimizer = tf.keras.optimizers.SGD(1e-3), loss = tf.keras.losses.categorical_crossentropy, metrics = ["accuracy"])
            model.fit(trains, epochs = 50, validation_data = (test_images, test_labels), verbose = True)
            
            end = time.time()
            
            loss, accuracy = model.evaluate(tests)
            
            print(f"Loss now is {loss}, accuracy = {accuracy}")
            print(f"Run numer #{i} times, seconds consumed: {end - start}")
    
if __name__ == "__main__":
    
    train()
