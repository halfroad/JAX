import tensorflow_datasets as tfds
import jax
import jax.example_libraries.optimizers
import sys

sys.path.append("../50/")

import ResNet

def setup():
    
    (trains, tests), metas = tfds.load(name = "cifar100", data_dir = "/tmp/", split = [tfds.Split.TRAIN, tfds.Split.TEST], with_info = True, batch_size = -1)
    
    trains = tfds.as_numpy(trains)
    tests = tfds.as_numpy(tests)
    
    train_images, train_labels = trains["image"], trains["label"]
    test_images, test_labels = tests["image"], tests["label"]
    
    train_images = train_images.astype(jax.numpy.float32) / 225.0
    test_images = test_images.astype(jax.numpy.float32) / 225.0
    
    num_classes = 100
    
    #train_images = jax.numpy.reshape(train_images, [-1, 3, 32, 32])
    #test_images = jax.numpy.reshape(test_images, [-1, 3, 32, 32])
    
    # train_images = jax.numpy.transpose(train_images, [0, 2, 3, 1])
    # test_images = jax.numpy.transpose(test_images, [0, 2, 3, 1])
    
    train_labels = jax.nn.one_hot(train_labels, num_classes = num_classes)
    test_labels = jax.nn.one_hot(test_labels, num_classes = num_classes)
    
    init_random_params_function, predict_function = ResNet.ResNet50(100)
    
    prng = jax.random.PRNGKey(10)
    inputs_shape = [-1, 32, 32, 3]
    batch_size = 100
    total_number = 10000
    epochs = 2
    
    return (train_images, train_labels), (test_images, test_labels), (num_classes, prng, inputs_shape, batch_size, total_number, epochs, init_random_params_function, predict_function)

def verify_accuracy(params, batch):
    
    inputs, targets = batch
    
    predictions = predict(params, inputs)
    _classes = jax.numpy.argmax(predictions, axis = 1)
    targets = jax.numpy.argmax(targets, axis = 1)
    
    return jax.numpy.sum(_classes == targets)


# Loss Function
def loss_function(params, batch, predict_function):
    
    inputs, targets = batch
    
    losses = -targets * predict_function(params, inputs)
    losses = jax.numpy.sum(losses, axis = 1)
    losses = jax.numpy.mean(losses)
    
    return losses

# Params Update Function
def update_function(i, optimizer_state, get_params_function, optimizer_update_function, batch, predict_function):
    
    params = get_params_function(optimizer_state)
    
    loss_function_grad = jax.grad(loss_function)
    gradients = loss_function_grad(params, batch, predict_function)
    
    return optimizer_update_function(i, gradients, optimizer_state)

def train():
    
    (train_images, train_labels), (test_images, test_labels), (num_classes, prng, inputs_shape, batch_size, total_number, epochs, init_random_params_function, predict_function) = setup()
    
    print((train_images.shape, train_labels.shape), (test_images.shape, test_labels.shape), (num_classes, prng, inputs_shape, batch_size, total_number, epochs, init_random_params_function, predict_function))
    
    # 100 ((50000, 32, 32, 3), (50000, 100)) ((10000, 32, 32, 3), (10000, 100))
    # 100 ((50000, 32, 32, 3), (50000,)) ((10000, 32, 32, 3), (10000,))
    
    optimizer_init, optimizer_update, get_params = jax.example_libraries.optimizers.adam(step_size = 1e-4)
    _, init_params = init_random_params_function(prng, inputs_shape)
    optimizer_state = optimizer_init(init_params)
    
    batch_number = total_number // batch_size
    
    for i in range(epochs):
        
        for j in range(batch_number):
        
            start = j * batch_size
            end = (j + 1) * batch_size
            
            batch = (train_images[start: end], train_labels[start: end])
            # def update_function(i, optimizer_state, get_params_function, optimizer_update_function, batch, predict_function):
            optimizer_state = update_function(i, optimizer_state, get_params, optimizer_update, batch, predict_function)
            
            if (j + 1) % 10 == 0:
                
                params = get_params(optimizer_state)
                # def loss_function(params, batch, predict_function):
                losses = loss_function(params, batch, predict_function)
                
                print("Losses = ", losses)
                
    params = get_params(optimizer_state)
    
    accuracies = []
    predictions = 0.0
    
    for i in range(epochs):
        
        start = i * batch_size
        end = (i + 1) * batch_size
        
        batch = (test_images[start: end], test_labels[start: end])
        
        predictions += verify_accuracy(params, batch)
        
    accuracies.append(predictions / float(total_number))
    
    print("Accuracies = ", {accuracies})
        
def main():
    
    train()
    
if __name__ == "__main__":
    
    main()
