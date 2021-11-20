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


'''
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
'''


class Neuron():
    def __init__(self, num_connections):
        self.output = 0
        self.input = 0
        self.weights = np.empty(num_connections)
        self.num_weights = num_connections
        self.gradients = np.empty(0)


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
    def __init__(self, num_inputs, layers, num_outputs, activation_function, final_function, batch_size):
        self.num_inputs = num_inputs
        self.input_layer = np.empty(0)
        self.output_layer = []
        self.num_layers = len(layers)
        self.layers = layers
        self.num_outputs = num_outputs
        self.outputs = np.empty(num_outputs)
        self.expected = np.empty(num_outputs)
        self.errors = np.empty(num_outputs)
        self.activation_function = activation_function
        self.final_function = final_function
        self.batch_size = batch_size
        self.batch_count = 0

    '''
    def initialize_input_layer(self, inputs):
        for i in range(self.num_inputs):
            self.input_layer.append(inputs[i])
    '''

    def initialize_layers(self, initial_weight):
        for i in range(self.num_layers):
            for j in range(self.layers[i].num_neurons):
                if i == 0:
                    for k in range(self.num_inputs):
                        self.layers[i].neurons[j].weights[k] = initial_weight
                else:
                    for k in range(self.layers[i-1].num_neurons):
                        self.layers[i].neurons[j].weights[k] = initial_weight

    def initialize_expected(self, expected):
        self.expected = expected

    def forward_feed(self, inputs):

        # input layer
        for input in inputs:
            self.input_layer = np.append(self.input_layer, input)

        for i in self.num_layers:
            if i == 0:
                for neuron in self.layers[i]:
                    input = np.matmul(neuron.weights, self.input_layer)
                    neuron.input = input
                    vectorized_activation_function = np.vectorize(
                        self.activation_function)
                    neuron.output = vectorized_activation_function(input)
            else:
                for j in self.layers[i].num_neurons:
                    input = 0
                    previous_layer = self.layers[i-1]
                    neuron = self.layers[i].neurons[j]
                    for k in range(previous_layer.num_neurons):
                        input += neuron.weights[k] * \
                            previous_layer.neurons[k].output
                    neuron.input = input
                    if i != (self.num_layers-1):
                        vectorized_activation_function = np.vectorize(
                            self.activation_function)
                        neuron.output = vectorized_activation_function(input)
                    # output layer
                    else:
                        vectorized_final_function = np.vectorize(
                            self.final_function)
                        neuron.output = vectorized_final_function(input)

    # calculate output portion of gradient
    # recursively calculate layer portion of gradient
    def calculate_gradients(self):
        # list iterates through the layers backwards from top to bottom
        gradients = np.empty(0)
        # calculate for one layer.
        dl_dw = 1  # identity multiplication
        dl_dyhat = 0
        dyhat_doutput = 0
        for i in reversed(range(self.num_layers)):
            # output portion
            current_layer = self.layers[i]
            output_layer = self.layers[-1]
            for j in range(output_layer.num_neurons):
                neuron = output_layer.num_neurons[j]
                for k in neuron.num_weights:
                    weight = neuron.weights[k]
                    dl_dyhat = -(self.expected[k] - neuron.output)
                    dyhat_doutput = neuron.output * \
                        (1 - neuron.output)
                    dl_dw *= (dl_dyhat * dyhat_doutput)

            # layer portion
            num_reps = self.num_layers - i
            # if the weight is in between, you don't take the derivative
            # store the gradient when it's calculated
            # assign the list to the neuron
            while num_reps != -1:
                layer = self.layers[num_reps]
                for neuron in layer.neurons:
                    doutput_dinput = ((1 + math.exp(-1 * neuron.input))
                                      ** -2) * math.exp(-1 * neuron.input)
                    # previous layer is the input layer
                    if num_reps == 0:
                        for neuron_previous in range(self.input_layer):
                            dinput_dw = neuron_previous.output
                            dl_dw *= (doutput_dinput * dinput_dw)
                            gradients = np.append(gradients, dl_dw)
                    else:
                        for neuron_previous in range(self.layers[num_reps-1]):
                            dinput_dw = neuron_previous.output
                            dl_dw *= (doutput_dinput * dinput_dw)

        '''
        for neuron, error in zip(self.layers[-1].neurons, self.errors):
            for weight in neuron.weights:
                # old weight - learning_parameter * delta_e
                weight = update_weight(
                    weight, final_output, neuron.output, expected, learning_rate)
                print("new weight:", weight)

        depth = 0
        while depth < (self.num_hidden_layers + 1):
            for neuron, final_output, expected in zip(output_layer.neurons, final_outputs, classifications):
                print("neuron, expected:", expected)
                for weight in neuron.weights:
                    # old weight - learning_parameter * delta_e
                    weight = update_weight(
                        weight, final_output, neuron.output, expected, learning_rate)
                    print("new weight:", weight)

        dl_dw11 = 0
        dl_dyhat = -(expected - final_output)
        dyhat_doutput = final_output * (1 - final_output)
        doutput_dinput = ((1 + math.exp(-1 * neuron_output))
                          ** -2) * math.exp(-1 * neuron_output)
        dinput_dw11 = hidden_layer_1.neurons[0].weights[0]
        dl_dw11 = dl_dyhat * dyhat_doutput * doutput_dinput * dinput_dw11

        print(dl_dw11)
        weight = weight - learning_rate * dl_dw11
        return weight
        '''

    # average the gradients across each weight;
    def back_propagate(self, learning_rate):
        pass
        # print(dl_dw11)
        #weight = weight - learning_rate * dl_dw11
    '''
    def initialize_output_layer(self, num_inputs, initial_weight):
        for i in range(self.num_outputs):
            output_neuron = Neuron(num_inputs)
            self.output_layer.append(output_neuron)

        for i in range(self.num_outputs):
            for j in range(num_inputs):
                self.output_layer[i].weights[j] = initial_weight
    '''

    # def update_weights(self, expected, learning_rate):
    def update_weights(self, final_outputs, expected_outputs, learning_rate):


if __name__ == "__main__":

    images = []
    labels = []
    images, labels = read_input("train_image1.csv", "train_label.csv")
    batch_size = 2
    batches = divide_into_batches(images, labels, batch_size)

    layers = []
    num_inputs = 28*28
    layer_1_num_neurons = 64
    num_outputs = 10
    layer_1 = Layer(layer_1_num_neurons)
    output_layer = Layer(num_outputs)
    layers.append(layer_1)
    layers.append(output_layer)
    neuralNetwork = NeuralNetwork(
        num_inputs, layers, sigmoid_function, softmax_function)

    # each set in the batch comes with an image and a label.
    # update weights after a batch
    for batch in batches:
        inputs = batch[0][0]
        label = int(batch[0][1])
        expected = np.zeros(10)
        expected[label] = 1
        neuralNetwork.forward_feed(inputs)
        neuralNetwork.back_propagate()

    '''
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
            weight = update_weight(weight, final_output,
                                   neuron.output, expected, learning_rate)
            print("new weight:", weight)

    for neuron_input, neuron_output, final_output, expected in zip(hidden_layer_1.neurons, output_layer.neurons, final_outputs, classifications):
        print("neuron, expected:", expected)
        for weight in neuron_input.weights:
            # old weight - learning_parameter * delta_e
            weight = update_weight(weight, final_output,
                                   neuron.output, expected, learning_rate)
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
