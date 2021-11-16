import numpy as np
from io import StringIO
import random

# take in input
# each input node represents a pixel


def read_input(image_file, label_file):
    images = np.genfromtxt(image_file, delimiter=",")
    labels = np.genfromtxt(label_file, delimiter="\n")
    return images, labels


class Neuron():
    def __init__(self):
        # to make sure that inputs are getting through
        # input
        self.weights = []
        self.output = 0


'''
class Layer():
    def __init__(self, **kwargs):
        self.neurons = np.empty(kwargs.get("num_neurons", 0))
        self.layer_order = kwargs.get("layer_order", 0)

    def load_inputs(self, inputs):
        if self.layer_order == 0:
            for index in range(len(inputs)):
                self.neurons[index] = inputs[index]
        elif self.layer_order == 1:
            # multiply weight of each connection by input
            input_final = 0
            for index in range(len(inputs)):
'''


class HiddenLayer():
    def __init__(self, num_inputs, num_neurons):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.weight_matrix = np.empty([
            self.num_neurons, self.num_inputs
        ])
        self.outputs = np.empty(num_neurons)

    def calculate_weight_matrix(self):
        for neuron in range(self.num_neurons):
            for input in range(self.num_inputs):
                self.weight_matrix[neuron][input] = 1

    def calculate_outputs(self, inputs):
        self.outputs = np.matmul(self.weight_matrix, inputs)

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
    # input layer
    inputs = images[0]

    
    num_neurons_hidden_layer_1 = 4
    hidden_layer_1 = HiddenLayer(num_neurons_hidden_layer_1, len(inputs))
    hidden_layer_1.calculate_weight_matrix()
    hidden_layer_1.calculate_outputs(inputs)

    num_neurons_hidden_layer_2 = 5
    hidden_layer_2 = HiddenLayer(num_neurons_hidden_layer_2, num_neurons_hidden_layer_1)
    hidden_layer_2.calculate_weight_matrix()
    hidden_layer_2.calculate_outputs(hidden_layer_1.output)

    num_classes = 10
    output_layer = HiddenLayer(num_classes)
    output_layer.calculate_weight_matrix()
    hidden_layer_2.calculate_outputs(hidden_layer_2.output)
    print(output_layer.output)
    

    # main idea: cross product of inputs times weights

    # hidden layer 1
    # matrix of weights
    # matrix of inputs
    # row of weights x column of input = output
    
    '''
    num_inputs_hidden_1 = len(inputs)
    num_neurons_hidden_1 = 10
    weight_matrix_input_to_hidden_1 = np.empty(
        [num_neurons_hidden_1, num_inputs_hidden_1])
    for neuron in range(num_neurons_hidden_1):
        for input in range(num_inputs_hidden_1):
            weight_matrix_input_to_hidden_1[neuron][input] = 1

    hidden_layer_1 = np.matmul(weight_matrix_input_to_hidden_1, inputs)
    
    # hidden layer 2
    num_inputs_hidden_2 = len(hidden_layer_1)
    num_neurons_hidden_2 = 10
    weight_matrix_hidden_1_to_hidden_2 = np.empty(
        [num_neurons_hidden_2, num_inputs_hidden_2])
    for neuron in range(num_neurons_hidden_2):
        for input in range(num_inputs_hidden_2):
            weight_matrix_hidden_1_to_hidden_2[neuron][input] = 2

    hidden_layer_2 = np.matmul(
        weight_matrix_hidden_1_to_hidden_2, hidden_layer_1)

    # output layer
    num_inputs_output = len(hidden_layer_2)
    num_classes_output = 10
    weight_matrix_hidden_2_to_output = np.empty(
        [num_classes_output, num_inputs_output])
    for classification in range(num_classes_output):
        for input in range(num_inputs_output):
            weight_matrix_hidden_2_to_output[classification][input] = 3

    classes = np.matmul(weight_matrix_hidden_2_to_output, hidden_layer_2)
    '''
