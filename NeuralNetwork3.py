import numpy as np
from io import StringIO
import random
import math
import pdb
import time
from csv import DictReader
import sys

# take in input
# each input node represents a pixel

# if you are not getting near perfect accuracy, you likely have a problem in your code.
# don't need to implement bias.
#randomize initial weights




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
    for value in x:
        if value < 0:
            value = 1 - 1/(1 + math.exp(value))
        else:
            value = 1/(1 + math.exp(-value))
    return x

def sigmoid_derivative(input):
    return (sigmoid_function(-input) ** -2) * np.exp(-input)

def softmax_function(outputs):

    outputs = np.exp(outputs)

    norm_values = outputs / np.sum(outputs)

    return norm_values

def softmax_derivative(values):
    calculated_values = []
    for value in values:
        calculated_values.append(value * (1 - value))
    return calculated_values

def forward_feed(inputs, nn):
    nn_state = {}

    nn_state['o0'] = inputs
    
    nn_state['z1'] = np.dot(nn['w0'], nn_state['o0'])
    nn_state['o1'] = sigmoid_function(nn_state['z1'])

    nn_state['z2'] = np.dot(nn['w1'], nn_state['o1'])
    nn_state['o2'] = softmax_function(nn_state['z2'])

    return nn_state

def calculate_gradients(nn_state, expected):
    for i in reversed(range(2)):
        if i == 1:
            #output layer
            nn_state['g2'] = []
            for j in range(10):
                gradient = nn_state['o1'][j] * sigmoid_derivative(nn_state['z2'][j]) * -(expected[j] - nn_state['o2'][j])
                nn_state['g2'].append(gradient)
        else:
            nn_state['g1'] = []
            for j in range(64):
                gradient = nn_state['o0'][j] * sigmoid_derivative(nn_state['z1'][j])
                for k in range(10):
                    gradient *= nn_state['o1'][k] * sigmoid_derivative(nn_state['z2'][k]) * -(expected[k] - nn_state['o2'][k])
                nn_state['g1'].append(gradient)

def calculate_new_weight(old_weight, learning_rate, gradient):
    return old_weight - learning_rate * gradient

def get_cost(outputs, expected_values):
    costs = []
    for output, expected in zip(outputs, expected_values):
        costs.append((1/2) * ((expected - output) ** 2))
    return sum(costs)

epochs = 100
learning_rate = 0.001
num_inputs = 28*28
num_outputs = 10
batch_size = 500
layer_sizes = [num_inputs, 64, num_outputs]
num_layers = 3
nn = initialize_network(layer_sizes)
images = np.genfromtxt(sys.argv[1], delimiter=",")
labels = np.genfromtxt(sys.argv[2], delimiter="\n")
gradients_g1 = np.zeros(64)
gradients_g2 = np.zeros(10)
print("TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING")
for e in range(epochs):
    print('epoch', e)
    start_time = time.time()
    samples = 10000
    cost = 0
    num_correct = 0
    for i in range(samples):

                input = images[i]
                label = labels[i]
                expected_values = np.zeros(num_outputs)
                expected_values[int(label)] = 1
                nn_state = forward_feed(input, nn)
                nn_state = calculate_gradients(nn_state, expected_values)
                gradients_g1 = [sum(x) for x in zip(gradients_g1, nn_state['g1'])]
                gradients_g2 = [sum(x) for x in zip(gradients_g2, nn_state['g2'])]
                if i+1 % batch_size == 0:
                    #update weights
                    for weight, gradient in zip(nn['w0'], nn_state['g1']):
                        weight = weight - learning_rate * gradient
                        gradients_g1 = np.zeros(64)
                    for weight, gradient in zip(nn['w1'], nn_state['g2']):
                        weight = weight - learning_rate * gradient
                        gradients_g2 = np.zeros(10)
                cost += get_cost(nn_state['o2'], expected_values)

                #nn['w0'] -= learning_rate * state['D0']
                #nn['w1'] -= learning_rate * state['D1']

                if np.argmax(nn_state['o2']) == np.argmax(expected_values):
                    num_correct += 1

    print("time:", time.time() - start_time)

    cost /= samples
    accuracy = num_correct / samples
    print('cost:', cost, 'accuracy:', accuracy)


