import numpy as np
import time
import sys
#import matplotlib.pyplot as pyplot

# take in input
# each input node represents a pixel

# if you are not getting near perfect accuracy, you likely have a problem in your code.
# don't need to implement bias.
# randomize initial weights

# forward pass for a batch
# aggregate errors and back propagate once

#average of time range and accuracy (range, well above 90)
#change hyper parameter by script

start_time = time.time()




def initialize_network(sizes, square_root_factor):
    network = {}
    for i, size in zip(range(len(sizes))[:len(sizes)-1], sizes[:len(sizes)-1]):
        network['w' + str(i)] = np.random.randn(sizes[i+1], size+1) * np.sqrt(float(square_root_factor) / sizes[i+1])
    return network

num_inputs = 784
num_outputs = 10
hidden_layer_1_num_nodes = 100
hidden_layer_2_num_nodes = 75
hidden_layer_3_num_nodes = 50
square_root_factor = 4
layer_sizes = [num_inputs, hidden_layer_1_num_nodes, hidden_layer_2_num_nodes, hidden_layer_3_num_nodes, num_outputs]
nn = initialize_network(layer_sizes, square_root_factor)


def sigmoid(x, derivative=False):
    if derivative:
        return np.exp(-x) / ((np.exp(-x) + 1) ** 2)
    else:
        return 1 / (1 + np.exp(-x))


def softmax(x, derivative=False):
    exp_shifted = np.exp(x - x.max())
    if derivative:
        return exp_shifted / np.sum(exp_shifted, axis=0) * (1 - exp_shifted / np.sum(exp_shifted, axis=0))
    else:
        return exp_shifted / np.sum(exp_shifted, axis=0)



biases = [1, 1, 1, 1]

def forward_feed(inputs, biases):
    
    nn_state = {}

    for i in range(len(layer_sizes)):
        if i == 0:
            nn_state['o0'] = inputs
            # bias
            nn_state['o0'] = np.append(nn_state['o0'], biases[i])
        elif i < len(layer_sizes)-1:
            nn_state['z' + str(i)] = np.matmul(nn['w' + str(i-1)], nn_state['o' + str(i-1)])
            nn_state['o' + str(i)] = sigmoid(nn_state['z' + str(i)], False)
            nn_state['o' + str(i)] = np.append(nn_state['o' + str(i)], biases[i])
        else:
            nn_state['z' + str(i)] = np.matmul(nn['w' + str(i-1)], nn_state['o' + str(i-1)])
            nn_state['o' + str(i)] = softmax(nn_state['z' + str(i)], False)

    return nn_state


def calculate_gradients(inputs, biases, expected):
    nn_state = forward_feed(inputs, biases)

    for i in reversed(range(len(layer_sizes))):
        if i == len(layer_sizes)-1:
            nn_state['g' + str(i)] = nn_state['o' + str(i)] - expected
            nn_state['D' + str(i-1)] = np.outer(nn_state['g' + str(i)], nn_state['o' + str(i-1)])
        elif i == len(layer_sizes)-2:
            nn_state['g' + str(i)] = np.matmul(nn_state['g' + str(i+1)], nn['w' + str(i)][:, 0:layer_sizes[i]]) * softmax(nn_state['z' + str(i)], derivative=True)
            nn_state['g' + str(i)] = np.append(nn_state['g' + str(i)], np.matmul(nn_state['g' + str(i+1)], nn['w' + str(i)][:, [layer_sizes[i]]] * softmax(nn_state['o' + str(i)][layer_sizes[i]], derivative=True)))
            nn_state['D' + str(i-1)] = np.outer(nn_state['g' + str(i)][0:layer_sizes[i]], nn_state['o' + str(i-1)])
        elif i > 0:
            layer_size_curr = layer_sizes[i]
            layer_size_prev = layer_sizes[i+1]
            nn_state['g' + str(i)] = np.matmul(nn_state['g' + str(i+1)][0:layer_size_prev], nn['w' + str(i)][0:layer_size_prev, 0:layer_size_curr]) * sigmoid(nn_state['z' + str(i)], derivative=True)
            nn_state['g' + str(i)] = np.append(nn_state['g' + str(i)], np.matmul(nn_state['g' + str(i+1)][0:layer_size_prev], nn['w' + str(i)][0:layer_size_prev, [layer_size_curr]] * sigmoid(nn_state['o' + str(i)][layer_size_curr], derivative=True)))
            nn_state['D' + str(i-1)] = np.outer(nn_state['g' + str(i)][0:layer_size_curr], nn_state['o' + str(i-1)])
    return nn_state
    


epochs = 50
learning_rate = 0.2
learning_adjust_epoch = 22
learning_adjust_rate = 0.002
images = np.genfromtxt("./train_image.csv", delimiter=",")
labels = np.genfromtxt("./train_label.csv", delimiter="\n")
accuracies = []

print("learning_rate", learning_rate, "batch_size SGD", "layer_sizes",
      layer_sizes, "biases", biases, "learning adjust epoch", learning_adjust_epoch, 
      "learning adjust rate", learning_adjust_rate, "epochs", epochs)
for e in range(epochs):
    #stabilize because accuracy tends to go above threshold at this point, and drop if not adjusted
    if e == learning_adjust_epoch:
        learning_adjust_rate = learning_adjust_rate
    print('epoch', e)
    start_time = time.time()
    cost = 0
    num_correct = 0
    num_samples = 0
    for i in range(10000):
        input = images[i]
        # normalize input
        input = (input / 255).astype('float32')
        label = labels[i]
        expected_values = np.zeros(num_outputs)
        #divide input by 10
        expected_values[int(label)] = 1
        nn_state = calculate_gradients(input, biases, expected_values)
        for i in range(len(layer_sizes)-1):
            nn['w' + str(i)] -= learning_rate * nn_state['D' + str(i)]

        if np.argmax(nn_state['o' + str(len(layer_sizes)-1)]) == np.argmax(expected_values):
            num_correct += 1
        num_samples += 1

    accuracy = num_correct / 10000
    accuracies.append(accuracy)
    print('accuracy:', accuracy)

    
end_time = time.time()
print(end_time - start_time, "seconds")

images_test = np.genfromtxt(sys.argv[3], delimiter=",")
#normalize the data
predictions = []
for input in images_test:
    input = (input / 255).astype('float32')
    nn_state = forward_feed(input)
    predictions.append(np.argmax(nn_state['o' + len(layer_sizes)]))

answers = np.genfromtxt(sys.argv[3], delimiter=",")
np.savetxt("test_predictions.csv", predictions, delimiter=",")
num_correct_test = 0
for i in range(10000):
    if answers[i] == predictions[i]:
        num_correct_test += 1
print("accuracy on test set:", num_correct_test / 10000)
