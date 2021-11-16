import numpy
from io import StringIO
import random

# take in input
# each input node represents a pixel


def read_input(image_file, label_file):
    images = numpy.genfromtxt(image_file, delimiter=",")
    labels = numpy.genfromtxt(label_file, delimiter="\n")
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
        self.neurons = numpy.empty(kwargs.get("num_neurons", 0))
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
    def __init__(self, neurons, inputs):
        self.neurons = []
        self.weight_matrix = numpy.empty(len(neurons), len(inputs))
        self.input_matrix = numpy.empty(len(inputs))
        self.output_matrix = numpy.empty(len(neurons))

    def calculate_output(self, inputs):
        '''
        num_inputs = len(inputs)
        for neuron in self.num_neurons:
            for input_index in range(num_inputs):
                self.weight_matrix[neuron][input_index] = 2
        '''

        self.output = numpy.matmul(self.weight_matrix, inputs)


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

    '''
    num_neurons_hidden_layer_1 = 10
    hidden_layer_1 = HiddenLayer(num_neurons_hidden_layer_1)
    hidden_layer_1.calculate_output(inputs)

    num_neurons_hidden_layer_2 = 15
    hidden_layer_2 = HiddenLayer(num_neurons_hidden_layer_2)
    hidden_layer_2.calculate_output(hidden_layer_1.output)

    num_classes = 10
    output_layer = HiddenLayer(num_classes)
    hidden_layer_2.calculate_output(hidden_layer_2.output)
    '''

    # main idea: cross product of inputs times weights

    # hidden layer 1
    # matrix of weights
    # matrix of inputs
    # row of weights x column of input = output
    '''
    num_inputs_hidden_1 = len(inputs)
    num_neurons_hidden_1 = 10
    weight_matrix_input_to_hidden_1 = numpy.empty(
        [num_neurons_hidden_1, num_inputs_hidden_1])
    for neuron in range(num_neurons_hidden_1):
        for input in range(num_inputs_hidden_1):
            weight_matrix_input_to_hidden_1[neuron][input] = 1

    hidden_layer_1 = numpy.matmul(weight_matrix_input_to_hidden_1, inputs)
    


    # hidden layer 2
    num_inputs_hidden_2 = len(hidden_layer_1)
    num_neurons_hidden_2 = 10
    weight_matrix_hidden_1_to_hidden_2 = numpy.empty(
        [num_neurons_hidden_2, num_inputs_hidden_2])
    for neuron in range(num_neurons_hidden_2):
        for input in range(num_inputs_hidden_2):
            weight_matrix_hidden_1_to_hidden_2[neuron][input] = 2

    hidden_layer_2 = numpy.matmul(
        weight_matrix_hidden_1_to_hidden_2, hidden_layer_1)

    # output layer
    num_inputs_output = len(hidden_layer_2)
    num_classes_output = 10
    weight_matrix_hidden_2_to_output = numpy.empty(
        [num_classes_output, num_inputs_output])
    for classification in range(num_classes_output):
        for input in range(num_inputs_output):
            weight_matrix_hidden_2_to_output[classification][input] = 3

    classes = numpy.matmul(weight_matrix_hidden_2_to_output, hidden_layer_2)
    '''
