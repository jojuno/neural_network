import numpy as np
from io import StringIO
import random
import math
import pdb
import time
from csv import reader
import sys

# take in input
# each input node represents a pixel

# if you are not getting near perfect accuracy, you likely have a problem in your code.
# don't need to implement bias.
#randomize initial weights


images = np.genfromtxt(sys.argv[1], delimiter=",")
labels = np.genfromtxt(sys.argv[2], delimiter="\n")

def initialize_network(sizes):
    input_layer = sizes[0]
    hidden_layer_1 = sizes[1]
    output_layer = sizes[2]
    network = {
        #dot after number: treat it like a float
        'w0': np.random.randn(hidden_layer_1, input_layer) * np.sqrt(1. / hidden_layer_1),
        'w1': np.random.randn(output_layer, hidden_layer_1) * np.sqrt(1. / output_layer)
    }
    return network


def sigmoid_function(x):
    return 1 / (1 + math.e ** (-x))

def softmax_function(outputs):

    exp_values = np.empty(0)
    for output in outputs:
        exp_values = np.append(exp_values, math.exp(output))

    norm_values = exp_values / np.sum(exp_values)

    return norm_values

def softmax_derivative(values):
    calculated_values = []
    for value in values:
        calculated_values.append(value * (1 - value))
    return calculated_values

num_inputs = 28*28
num_outputs = 10
layer_sizes = [num_inputs, 64, num_outputs]
nn = initialize_network(layer_sizes)


#return: outputs
def forward_feed(inputs):

    #input to first layer
    state = {}
    state['inputs'] = inputs

    #input to first layer 2
    state['i1'] = np.dot(nn['w0'], state['inputs'])
    state['o1'] = sigmoid_function(state['i1'])

    #first layer to output
    state['i2'] = np.dot(nn['w1'], state['o1'])
    state['o2'] = softmax_function(state['i2'])

    return state
    
def get_cost(outputs, expected_values):
    costs = []
    for output, expected in zip(outputs, expected_values):
        costs.append((1/2) * ((expected - output) ** 2))
    return sum(costs)

learning_rate = 0.001

def backward_propagate(inputs, expected_values):
    state = forward_feed(inputs)

    state['d2'] = state['o2'] - expected_values
    state['d1'] = np.dot(state['d2'], nn['w1'] * softmax_derivative(state['i1']))

    state['D1'] = np.outer(state['d2'], state['o1'])
    state['D0'] = np.outer(state['d1'], state['inputs'])

    return state
    

epochs = 200
learning_rate = 0.001
print("TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING")
for e in range(epochs):
    print('epoch', e)
    samples = 10
    cost_value = 0
    num_correct = 0
    for i in range(samples):
        expected_values = np.zeros(num_outputs)
        expected_values[int(labels[i])] = 1
        state = backward_propagate(images[i], expected_values)
        cost_value += get_cost(state['o2'], expected_values)

        nn['w0'] -= learning_rate * state['D0']
        nn['w1'] -= learning_rate * state['D1']

        if np.argmax(state['o2']) == np.argmax(expected_values):
            num_correct += 1

    cost_value = cost_value / samples
    accuracy = num_correct / samples
    print('cost:', cost_value, 'accuracy:', accuracy)


