import numpy as np
from io import StringIO
import random
import math

# take in input
# each input node represents a pixel

# if you are not getting near perfect accuracy, you likely have a problem in your code.
# don't need to implement bias.


def read_input(image_file, label_file):
    images = np.genfromtxt(image_file, delimiter=",")
    labels = np.genfromtxt(label_file, delimiter="\n")
    return images, labels


class NeuralNetwork():
    def __init__(self):
        self.input_layer = []
        self.hidden_layers = []
        self.output_layer = []


class Neuron():
    def __init__(self, num_connections):
        self.output = 0
        self.weights = np.empty(num_connections)


class HiddenLayer():
    def __init__(self, num_inputs, num_neurons, bias):
        self.num_inputs = num_inputs
        self.bias = bias
        self.num_neurons = num_neurons
        self.neurons = []
        self.weight_matrix = np.empty([
            self.num_neurons, self.num_inputs+1
        ])
        self.outputs = np.empty([num_neurons])

    def initialize_weight_matrix(self, initial_weight):
        for index in range(self.num_neurons):
            # add 1 for bias
            neuron = Neuron(self.num_inputs+1)
            for index_input in range(self.num_inputs):
                neuron.weights[index_input] = initial_weight
            self.neurons.append(neuron)

    def calculate_outputs(self, inputs, function):
        inputs = np.append(inputs, self.bias)
        for neuron, output in zip(self.neurons, self.outputs):
            output = np.matmul(neuron.weights, inputs)
            neuron.output = output
        # for the output layer, use the "softmax_function"
        if function != None:
            vfunc = np.vectorize(function)
            for output in self.outputs:
                vfunc(output)


def sigmoid_function(x):
    return 1 / (1 + math.e ** (-x))

# takes the whole vector. use for the output layer.


def softmax_function(outputs):

    exp_values = np.exp(outputs)

    norm_values = exp_values / np.sum(exp_values)

    return norm_values


def loss_function(desired_output, actual_output):
    return (1/2) * ((desired_output - actual_output) ** 2)
    # return desired_output - actual_output


def apply_loss_function(outputs, classifications, label):
    classifications[label] = 1
    def lambda_function(x, y): return loss_function(x, y)
    outputs = lambda_function(classifications, outputs)
    return outputs


def divide_into_batches(images, labels, batch_size):
    batch_count = 0
    training_size = len(images)
    batches = []
    batch = []
    for i in range(training_size):
        batch.append([images[i], labels[i]])
        batch_count += 1
        if batch_count == batch_size:
            batch_count = 0
            batches.append(batch)
            batch = []
    return batches


def back_propagate():
    pass


def calculate_delta_e():
    pass


def calculate_derivative_sigmoid(expected, actual, input_j, output_i):
    E = math.e
    del_e_over_del_output_j = -(expected - actual)
    del_output_j_over_del_input_j = (
        (1 + (E**(-input_j))) ** -2) * (E**(-input_j))
    del_input_j_over_del_weight_i_j = output_i
    print(del_e_over_del_output_j)
    print(del_output_j_over_del_input_j)
    print(del_input_j_over_del_weight_i_j)
    # return del_e_over_del_output_j * del_output_j_over_del_input_j * del_input_j_over_del_weight_i_j
    return -(expected - actual) * ((1 + (E**(-input_j))) ** -2) * (E**(-input_j)) * output_i

# how to determine learning rate?


def calculate_new_weight(old_weight, learning_rate, delta_e):
    return old_weight - learning_rate * delta_e


def softmax_derivative(x):
    return x * (1 - x)


if __name__ == "__main__":
    # feed forward network
    # take in a CSV file
    # feed it to two layers of hidden neurons
    # choose the number of neurons in each layer
    # feed it to the last layer
    # classify which number it is by taking the maximum of the neurons

    images = []
    labels = []
    images, labels = read_input("train_image1.csv", "train_label.csv")
    batch_size = 1
    batches = divide_into_batches(images, labels, batch_size)

    # input layer
    # each set in the batch comes with an image and a label.
    inputs = batches[0][0][0]
    label = int(batches[0][0][1])

    num_neurons_hidden_layer_1 = 4
    hidden_layer_1_bias = 1
    hidden_layer_1_initial_weight = 1
    hidden_layer_1 = HiddenLayer(
        len(inputs), num_neurons_hidden_layer_1, hidden_layer_1_bias)
    hidden_layer_1.initialize_weight_matrix(hidden_layer_1_initial_weight)
    hidden_layer_1.calculate_outputs(inputs, sigmoid_function)

    '''
    num_neurons_hidden_layer_2 = 5
    hidden_layer_2_bias = 1
    hidden_layer_2_initial_weight = 1
    hidden_layer_2 = HiddenLayer(
        num_neurons_hidden_layer_1, num_neurons_hidden_layer_2, hidden_layer_2_bias)
    hidden_layer_2.initialize_weight_matrix(hidden_layer_2_initial_weight)
    hidden_layer_2.calculate_outputs(hidden_layer_1.outputs, sigmoid_function)
    '''

    num_classes = 10
    output_layer_bias = 1
    output_layer_initial_weight = 1
    output_layer = HiddenLayer(
        num_neurons_hidden_layer_1, num_classes, output_layer_bias)
    output_layer.initialize_weight_matrix(output_layer_initial_weight)
    output_layer.calculate_outputs(hidden_layer_1.outputs, None)
    output_layer.outputs = softmax_function(output_layer.outputs)

    '''
    #last layer after softmax
    final_layer = HiddenLayer(1, num_classes, 0)
    final_layer.initialize_weight_matrix(output_layer_initial_weight)
    final_layer.calculate_outputs()
    '''

    classifications = np.zeros(10)
    # apply label (answer)
    classifications[label] = 1
    errors = apply_loss_function(
        output_layer.outputs, classifications, label)

    

    # back propagation
    
    dl_dh1 = dl_d
    for index in range(output_layer.num_neurons):
        output_layer.
