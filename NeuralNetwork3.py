import numpy as np
from io import StringIO
import random
import math
import pdb

# take in input
# each input node represents a pixel

# if you are not getting near perfect accuracy, you likely have a problem in your code.
# don't need to implement bias.


def read_input(image_file, label_file):
    images = np.genfromtxt(image_file, delimiter=",")
    labels = np.genfromtxt(label_file, delimiter="\n")
    return images, labels

'''
class NeuralNetwork():
    def __init__(self):
        self.input_layer = []
        self.hidden_layers = []
        self.output_layer = []

class HiddenLayer():
    def __init__(self, num_inputs, num_neurons):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.neurons = []
        self.weight_matrix = np.empty([
            self.num_neurons, self.num_inputs+1
        ])
        self.outputs = np.empty([num_neurons])

    def initialize_weight_matrix(self, initial_weight):
        for index in range(self.num_neurons):
            neuron = Neuron(self.num_inputs)
            for index_input in range(self.num_inputs):
                neuron.weights[index_input] = initial_weight
            self.neurons.append(neuron)

    def calculate_outputs(self, inputs, function):
        for neuron, output in zip(self.neurons, self.outputs):
            output = np.matmul(neuron.weights, inputs)
            neuron.output = output
        # for the output layer, use the "softmax_function"
        if function != None:
            vfunc = np.vectorize(function)
            for output in self.outputs:
                vfunc(output)
'''

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

def update_weight(weight, final_output, neuron_output, expected, learning_rate):
    dl_dw11 = 0
    dl_dyhat = -(expected - final_output)
    dyhat_doutput = final_output * (1 - final_output)
    doutput_dinput = ((1 + math.exp(-1 * neuron_output)) ** -2) * math.exp(-1 * neuron_output)
    dinput_dw11 = hidden_layer_1.neurons[0].weights[0]
    dl_dw11 = dl_dyhat * dyhat_doutput * doutput_dinput * dinput_dw11
    
    print(dl_dw11)
    weight = weight - learning_rate * dl_dw11
    return weight

class Neuron():
    def __init__(self, num_connections):
        self.output = 0
        self.input = 0
        self.weights = np.empty(num_connections)

class Layer():
    def __init__(self, num_neurons, num_previous_neurons, initial_weight):
        self.neurons = []
        self.num_neurons = num_neurons
        for i in range(num_neurons):
            neuron = Neuron(num_previous_neurons)
            for j in range(num_previous_neurons):
                neuron.weights[j] = initial_weight
            self.neurons.append(neuron)


class NeuralNetwork():
    def __init__(self, num_inputs, layers, num_outputs, activation_function, final_function):
        self.num_inputs = num_inputs
        self.input_layer = []
        self.output_layer = []
        self.num_layers = len(layers)
        self.layers = layers
        self.num_outputs = num_outputs
        self.activation_function = activation_function
        self.final_function = final_function

    def initialize_input_layer(self, inputs):
        for i in range(self.num_inputs):
            input_neuron = Neuron(0)
            input_neuron.input = inputs[i]
            self.input_layer.append(input_neuron)

    def initialize_layers(self, initial_weight):
        for i in range(self.num_layers):
            for j in range(self.layers[i].num_neurons):
                if i == 0:
                    for k in range(self.num_inputs):
                        self.layers[i].neurons[j].weights[k] = initial_weight
                else:
                    for k in range(self.layers[i-1].num_neurons):
                        self.layers[i].neurons[j].weights[k] = initial_weight

    def forward_feed(self):
        for i in self.num_layers:
            if i == 0:
                vectorized_activation_function = np.vectorize(self.activation_function)
                for neuron in self.layers[i]:
                    input = np.matmul(neuron.weights, self.input_layer)
                    neuron.input = input
                    neuron.output = vectorized_activation_function(input)
            elif i == (self.num_layers-1):
                vectorized_final_function = np.vectorize(self.final_function)
                for neuron in self.layers[i]:
                    input = np.matmul(neuron.weights, self.input_layer)
                    neuron.input = input
                    neuron.output = vectorized_final_function(input)

    def back_propagate(self):
        for i in reversed(range(self.num_layers)):
            layer = self.layers[i]
            for j in range(layer.num_neurons):


        for neuron, final_output, expected in zip(self.output_layer, final_outputs, classifications):
            print("neuron, expected:", expected)
            for weight in neuron.weights:
                # old weight - learning_parameter * delta_e
                weight = update_weight(weight, final_output, neuron.output, expected, learning_rate)
                print("new weight:", weight)

        depth = 0
        while depth < (self.num_hidden_layers + 1):
            for neuron, final_output, expected in zip(output_layer.neurons, final_outputs, classifications):
                print("neuron, expected:", expected)
                for weight in neuron.weights:
                    # old weight - learning_parameter * delta_e
                    weight = update_weight(weight, final_output, neuron.output, expected, learning_rate)
                    print("new weight:", weight)

        dl_dw11 = 0
        dl_dyhat = -(expected - final_output)
        dyhat_doutput = final_output * (1 - final_output)
        doutput_dinput = ((1 + math.exp(-1 * neuron_output)) ** -2) * math.exp(-1 * neuron_output)
        dinput_dw11 = hidden_layer_1.neurons[0].weights[0]
        dl_dw11 = dl_dyhat * dyhat_doutput * doutput_dinput * dinput_dw11
        
        print(dl_dw11)
        weight = weight - learning_rate * dl_dw11
        return weight



    '''
    def initialize_output_layer(self, num_inputs, initial_weight):
        for i in range(self.num_outputs):
            output_neuron = Neuron(num_inputs)
            self.output_layer.append(output_neuron)

        for i in range(self.num_outputs):
            for j in range(num_inputs):
                self.output_layer[i].weights[j] = initial_weight
    '''

    #def update_weights(self, expected, learning_rate):
    def update_weights(self, final_outputs, expected_outputs, learning_rate):

        

if __name__ == "__main__":

    images = []
    labels = []
    images, labels = read_input("train_image1.csv", "train_label.csv")
    batch_size = 1
    batches = divide_into_batches(images, labels, batch_size)

    # input layer
    # each set in the batch comes with an image and a label.
    # eventually, update weights after a batch; currently, it's over one sample.
    inputs = batches[0][0][0]
    label = int(batches[0][0][1])

    

    layers = []
    layer_1_num_neurons = 64
    layer_1 = Layer(layer_1_num_neurons)
    #hidden_layer_2_num_neurons = 32
    #hidden_layer_2 = Layer(hidden_layer_2_num_neurons)
    num_outputs = 10
    output_layer = Layer(num_outputs)
    #layers.append(hidden_layer_2)
    layers.append(layer_1)
    layers.append(output_layer)
    num_inputs = 28*28
    num_outputs = 10
    neuralNetwork = NeuralNetwork(num_inputs, layers, sigmoid_function, softmax_function)

    num_neurons_hidden_layer_1 = 4
    hidden_layer_1_initial_weight = 1
    hidden_layer_1 = HiddenLayer(
        len(inputs), num_neurons_hidden_layer_1)
    hidden_layer_1.initialize_weight_matrix(hidden_layer_1_initial_weight)
    hidden_layer_1.calculate_outputs(inputs, sigmoid_function)

    '''
    num_neurons_hidden_layer_2 = 5
    hidden_layer_2_initial_weight = 1
    hidden_layer_2 = HiddenLayer(
        num_neurons_hidden_layer_1, num_neurons_hidden_layer_2)
    hidden_layer_2.initialize_weight_matrix(hidden_layer_2_initial_weight)
    hidden_layer_2.calculate_outputs(hidden_layer_1.outputs, sigmoid_function)
    '''

    num_classes = 10
    output_layer_initial_weight = 1
    output_layer = HiddenLayer(
        num_neurons_hidden_layer_1, num_classes)
    output_layer.initialize_weight_matrix(output_layer_initial_weight)
    output_layer.calculate_outputs(hidden_layer_1.outputs, None)

    final_outputs = softmax_function(output_layer.outputs)

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
    learning_rate = 1
    for neuron, final_output, expected in zip(output_layer.neurons, final_outputs, classifications):
        print("neuron, expected:", expected)
        for weight in neuron.weights:
            # old weight - learning_parameter * delta_e
            weight = update_weight(weight, final_output, neuron.output, expected, learning_rate)
            print("new weight:", weight)

    for neuron_input, neuron_output, final_output, expected in zip(hidden_layer_1.neurons, output_layer.neurons, final_outputs, classifications):
        print("neuron, expected:", expected)
        for weight in neuron_input.weights:
            # old weight - learning_parameter * delta_e
            weight = update_weight(weight, final_output, neuron.output, expected, learning_rate)
            print("new weight:", weight)

'''
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
    hidden_layer_1_initial_weight = 1
    hidden_layer_1 = HiddenLayer(
        len(inputs), num_neurons_hidden_layer_1)
    hidden_layer_1.initialize_weight_matrix(hidden_layer_1_initial_weight)
    hidden_layer_1.calculate_outputs(inputs, sigmoid_function)

    
    num_neurons_hidden_layer_2 = 5
    hidden_layer_2_initial_weight = 1
    hidden_layer_2 = HiddenLayer(
        num_neurons_hidden_layer_1, num_neurons_hidden_layer_2)
    hidden_layer_2.initialize_weight_matrix(hidden_layer_2_initial_weight)
    hidden_layer_2.calculate_outputs(hidden_layer_1.outputs, sigmoid_function)
    

    num_classes = 10
    output_layer_initial_weight = 1
    output_layer = HiddenLayer(
        num_neurons_hidden_layer_1, num_classes)
    output_layer.initialize_weight_matrix(output_layer_initial_weight)
    output_layer.calculate_outputs(hidden_layer_1.outputs, None)

    final_outputs = softmax_function(output_layer.outputs)

    
    #last layer after softmax
    final_layer = HiddenLayer(1, num_classes, 0)
    final_layer.initialize_weight_matrix(output_layer_initial_weight)
    final_layer.calculate_outputs()
    

    classifications = np.zeros(10)
    # apply label (answer)
    classifications[label] = 1
    errors = apply_loss_function(
        output_layer.outputs, classifications, label)

    # back propagation
    learning_rate = 1
    for neuron, final_output, expected in zip(output_layer.neurons, final_outputs, classifications):
        print("neuron, expected:", expected)
        for weight in neuron.weights:
            # old weight - learning_parameter * delta_e
            weight = update_weight(weight, final_output, neuron.output, expected, learning_rate)
            print("new weight:", weight)

    for neuron_input, neuron_output, final_output, expected in zip(hidden_layer_1.neurons, output_layer.neurons, final_outputs, classifications):
        print("neuron, expected:", expected)
        for weight in neuron_input.weights:
            # old weight - learning_parameter * delta_e
            weight = update_weight(weight, final_output, neuron.output, expected, learning_rate)
            print("new weight:", weight)
'''

