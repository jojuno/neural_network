import numpy as np
from io import StringIO
import random
import math
import time
import csv
import sys
import pandas
import matplotlib.pyplot as pyplot

# take in input
# each input node represents a pixel

# if you are not getting near perfect accuracy, you likely have a problem in your code.
# don't need to implement bias.
# randomize initial weights

# forward pass for a batch
# aggregate errors and back propagate once


def initialize_network(sizes):
    network = {}
    layer_num = 0
    for i in range(len(sizes)):
        network['w' + str(i)] = np.random.randn(sizes[i+1],
                                                sizes[i]) * np.sqrt(4. / sizes[i+1])
    return network


num_inputs = 28*28
num_outputs = 10
layer_sizes = [num_inputs, 128, 64, num_outputs]
nn = initialize_network(layer_sizes)


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


def cross_entropy(o, y, derivative=False):
    if derivative:
        result = np.empty(len(o))
        for result_value, output, expected in zip(result, o, y):
            result_value = -((expected - output) / ((1-output) * output))
        return result
    else:
        c = np.dot(y, np.log(o)) + np.dot((1 - y), np.log(1 - o))
        return -c


bias_0 = 5
bias_1 = 3
bias_2 = 1
biases = [5, 3, 1]


def forward_feed(inputs, biases):
    nn_state = {}

    for i in range(len(layer_sizes)):
        if i == 0:
            nn_state['o0'] = inputs
            nn_state['o0'] = np.append(nn_state['o0'], biases[0])
        elif i != len(layer_sizes) - 1:
            nn_state['z' + str(i)] = np.matmul(nn['w' + str(i-1)], nn_state['o' + str(i-1)])
            nn_state['o' + str(i)] = sigmoid(nn_state['z' + str(i)], False)
            nn_state['o' + str(i)] = np.append(nn_state['o' + str(i)], biases[i])
        else:
            nn_state['z' + str(i)] = np.matmul(nn['w' + str(i)], nn_state['o' + str(i)])
            nn_state['o' + str(i)] = softmax(nn_state['z' + str(i)], False)

    return nn_state


def calculate_gradients(inputs, expected):
    nn_state = forward_feed(inputs)

    for i in reversed(range(len(layer_sizes))):
        if i == (len(layer_sizes) - 1):
            nn_state['g' + str(i)] = nn_state['o' + str(i)] - expected
        elif i == (len(layer_sizes) - 2):
            nn_state['g' + str(i)] = np.matmul(nn_state['g' + str(i+1)], nn['w' + str(i)][:, 0:layer_sizes[i]]) * softmax(nn_state['z' + str(i)], derivative=True)
            nn_state['g' + str(i)] = np.append(nn_state['g' + str(i)], np.matmul(nn_state['g' + str(i+1)], nn['w' + str(i)][:, [layer_sizes[i]]] * softmax(nn_state['o' + str(i)][layer_sizes[i]], derivative=True)))
        elif i > 0: 
            nn_state['g1'] = np.matmul(nn_state['g2'][0:64], nn['w1']
                                    [0:64, 0:128]) * sigmoid(nn_state['z1'], derivative=True)
            nn_state['g1'] = np.append(nn_state['g1'], np.matmul(
                nn_state['g2'][0:64], nn['w1'][0:64, [128]] * sigmoid(nn_state['o1'][128], derivative=True)))

    nn_state['D2'] = np.outer(nn_state['g3'], nn_state['o2'])
    nn_state['D1'] = np.outer(nn_state['g2'][0:64], nn_state['o1'])
    nn_state['D0'] = np.outer(nn_state['g1'][0:128], nn_state['o0'])

    return nn_state


def get_cost(outputs, expected_values):
    costs = []
    for output, expected in zip(outputs, expected_values):
        costs.append((1/2) * ((expected - output) ** 2))
    return -sum(costs)


epochs = 100
learning_rate = 0.005
batch_size = 1
#images = np.genfromtxt(sys.argv[1], delimiter=",")
images = np.genfromtxt("./train_image.csv", delimiter=",")
#labels = np.genfromtxt(sys.argv[2], delimiter="\n")
labels = np.genfromtxt("./train_label.csv", delimiter="\n")
print("TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING")
accuracies = []
samples = random.sample(range(60000), 10000)
nn_state_aggregation = {}

print("learning_rate", learning_rate, "batch_size", batch_size, "layer_sizes:",
      layer_sizes, "bias_0:", bias_0, "bias_1:", bias_1, "bias_2:", bias_2)
for e in range(epochs):
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
        expected_values[int(label)] = 1
        nn_state = calculate_gradients(input, expected_values)
        if num_samples % batch_size == 0:
            nn_state_aggregation = dict(nn_state)
        else:
            for value1, value2 in zip(nn_state_aggregation.values(), nn_state.values()):
                value1 += value2
        if (num_samples+1) % batch_size == 0:
            # update weights
            for value in nn_state_aggregation.values():
                value /= batch_size
            nn['w0'] -= learning_rate * nn_state_aggregation['D0']
            nn['w1'] -= learning_rate * nn_state_aggregation['D1']
            nn['w2'] -= learning_rate * nn_state_aggregation['D2']
            nn_state_aggregation = {}
            cost += cross_entropy(nn_state['o3'], expected_values)

        if np.argmax(nn_state['o3']) == np.argmax(expected_values):
            num_correct += 1
        num_samples += 1

    print("time:", time.time() - start_time)

    cost /= len(samples)
    accuracy = num_correct / len(samples)
    accuracies.append(accuracy)
    print('cost:', cost, 'accuracy:', accuracy)
pyplot.plot(accuracies)
pyplot.show()

#images_test = np.genfromtxt(sys.argv[3], delimiter=",")
images_test = np.genfromtxt("./test_image.csv", delimiter=",")
predictions = []
for input in images_test:
    nn_state = forward_feed(input)
    predictions.append(np.argmax(nn_state['o3']))

predictions = np.asarray([predictions])
np.savetxt("test_predictions.csv", predictions, delimiter=",")
