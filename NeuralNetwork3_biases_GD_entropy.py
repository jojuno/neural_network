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
    input_layer = sizes[0]
    hidden_layer_1 = sizes[1]
    hidden_layer_2 = sizes[2]
    hidden_layer_3 = sizes[3]
    output_layer = sizes[4]
    network = {
        # dot after number: treat it like a float
        # add 1 for biases
        'w0': np.random.randn(hidden_layer_1, input_layer + 1) * np.sqrt(1. / hidden_layer_1),
        'w1': np.random.randn(hidden_layer_2, hidden_layer_1 + 1) * np.sqrt(1. / hidden_layer_2),
        'w2': np.random.randn(hidden_layer_3, hidden_layer_2 + 1) * np.sqrt(1. / hidden_layer_3),
        'w3': np.random.randn(output_layer, hidden_layer_3 + 1) * np.sqrt(1. / output_layer)
    }
    return network


num_inputs = 28*28
num_outputs = 10
layer_sizes = [num_inputs, 128, 64, 32, num_outputs]
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


def cross_entropy_derivative(output, expected):
    return -((expected - output) / ((1-output) * output))


def cross_entropy(o, y, derivative=False):
    if derivative:
        results = np.empty(len(o))
        for result, output, expected in zip(results, o, y):
            result = -((expected - output) / ((1-output) * output))
        return results
    else:
        c = np.dot(y, np.log(o)) + np.dot((1 - y), np.log(1 - o))
        return -c


def forward_feed(inputs):
    nn_state = {}

    nn_state['o0'] = inputs
    # bias
    nn_state['o0'] = np.append(nn_state['o0'], 1)

    nn_state['z1'] = np.matmul(nn['w0'], nn_state['o0'])
    nn_state['o1'] = sigmoid(nn_state['z1'], False)
    nn_state['o1'] = np.append(nn_state['o1'], 1)

    nn_state['z2'] = np.matmul(nn['w1'], nn_state['o1'])
    nn_state['o2'] = sigmoid(nn_state['z2'], False)
    nn_state['o2'] = np.append(nn_state['o2'], 1)

    nn_state['z3'] = np.matmul(nn['w2'], nn_state['o2'])
    nn_state['o3'] = sigmoid(nn_state['z3'], False)
    nn_state['o3'] = np.append(nn_state['o3'], 1)

    nn_state['z4'] = np.matmul(nn['w3'], nn_state['o3'])
    nn_state['o4'] = softmax(nn_state['z4'], False)

    return nn_state


def calculate_gradients(inputs, expected):
    nn_state = forward_feed(inputs)

    nn_state['g4'] = nn_state['o4'] - expected
    #nn_state['g4'] = cross_entropy(nn_state['o4'], expected, derivative=True)
    nn_state['g3'] = np.matmul(
        nn_state['g4'], nn['w3'][:, 0:32]) * softmax(nn_state['z3'], derivative=True)
    nn_state['g3'] = np.append(nn_state['g3'], np.matmul(
        nn_state['g4'], nn['w3'][:, [32]] * softmax(nn_state['o3'][32], derivative=True)))
    nn_state['g2'] = np.matmul(nn_state['g3'][0:32], nn['w2']
                               [0:32, 0:64]) * sigmoid(nn_state['z2'], derivative=True)
    nn_state['g2'] = np.append(nn_state['g2'], np.matmul(
        nn_state['g3'][0:32], nn['w2'][0:32, [64]] * sigmoid(nn_state['o2'][64], derivative=True)))
    nn_state['g1'] = np.matmul(nn_state['g2'][0:64], nn['w1']
                               [0:64, 0:128]) * sigmoid(nn_state['z1'], derivative=True)
    nn_state['g1'] = np.append(nn_state['g1'], np.matmul(nn_state['g2'][0:64], nn['w1'][0:64, [
                               128]] * sigmoid(nn_state['o1'][128], derivative=True)))

    nn_state['D3'] = np.outer(nn_state['g4'], nn_state['o3'])
    nn_state['D2'] = np.outer(nn_state['g3'][0:32], nn_state['o2'])
    nn_state['D1'] = np.outer(nn_state['g2'][0:64], nn_state['o1'])
    nn_state['D0'] = np.outer(nn_state['g1'][0:128], nn_state['o0'])

    return nn_state


def get_cost(outputs, expected_values):
    costs = []
    for output, expected in zip(outputs, expected_values):
        costs.append((1/2) * ((expected - output) ** 2))
    return -sum(costs)


epochs = 200
learning_rate = 0.001
batch_size = 100
# images = np.genfromtxt(sys.argv[1], delimiter=",")
images = np.genfromtxt("./train_image.csv", delimiter=",")
# labels = np.genfromtxt(sys.argv[2], delimiter="\n")
labels = np.genfromtxt("./train_label.csv", delimiter="\n")
print("TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING")
accuracies = []

nn_state_aggregation = {}
for e in range(epochs):
    print('epoch', e)
    start_time = time.time()
    samples = random.sample(range(60000), 10000)
    cost = 0
    num_correct = 0
    num_samples = 0
    for i in samples:
        input = images[i]
        # normalize input
        input = (input / 255).astype('float32')
        label = labels[i]
        expected_values = np.zeros(num_outputs)
        expected_values[int(label)] = 1
        nn_state = calculate_gradients(input, expected_values)
        if num_samples == 0:
            nn_state_aggregation = dict(nn_state)
        else:
            for value1, value2 in zip(nn_state.values(), nn_state_aggregation.values()):
                value2 += value1
        if (num_samples+1) % batch_size == 0:
            # update weights
            for value in nn_state_aggregation.values():
                value /= batch_size
            nn['w0'] -= learning_rate * nn_state_aggregation['D0']
            nn['w1'] -= learning_rate * nn_state_aggregation['D1']
            nn['w2'] -= learning_rate * nn_state_aggregation['D2']
            nn['w3'] -= learning_rate * nn_state_aggregation['D3']
            nn_state_aggregation = {}
            num_samples = -1
        cost += cross_entropy(nn_state['o4'], expected_values)

        if np.argmax(nn_state['o4']) == np.argmax(expected_values):
            num_correct += 1
        num_samples += 1

    print("time:", time.time() - start_time)

    cost /= len(samples)
    accuracy = num_correct / len(samples)
    accuracies.append(accuracy)
    print('cost:', cost, 'accuracy:', accuracy)
    pyplot.plot(accuracies)
    pyplot.show()

# images_test = np.genfromtxt(sys.argv[3], delimiter=",")
images_test = np.genfromtxt("./test_image.csv", delimiter=",")
predictions = []
for input in images_test:
    nn_state = forward_feed(input)
    predictions.append(np.argmax(nn_state['o4']))

predictions = np.asarray([predictions])
np.savetxt("test_predictions.csv.", predictions, delimiter=",")
