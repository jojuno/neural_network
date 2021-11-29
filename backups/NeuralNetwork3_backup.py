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


def read_input(image_file, label_file):
    images = np.genfromtxt(image_file, delimiter=",")
    labels = np.genfromtxt(label_file, delimiter="\n")
    return images, labels


def sigmoid_function(x):
    return 1 / (1 + math.e ** (-x))

# takes the whole vector. use for the output layer.

def softmax_function(outputs):

    exp_values = np.empty(0)
    for output in outputs:
        exp_values = np.append(exp_values, math.exp(output))

    norm_values = exp_values / np.sum(exp_values)

    return norm_values

'''
def softmax_function(output, outputs):

    exp_values = np.empty(0)
    for output_array in outputs:
        exp_values = np.append(exp_values, math.exp(output_array))

    norm_value = math.exp(output) / np.sum(exp_values)

    return norm_value
'''

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


def calculate_derivative_sigmoid(input):
    return ((1 + math.exp(-input)) ** -2) * math.exp(-input)

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

'''
class Neuron():
    def __init__(self, num_connections):
        self.output = 0
        self.input = 0
        self.weights = np.empty(num_connections)
        self.num_weights = num_connections
        #append lists
        self.gradient_table = np.empty(shape = (num_connections))


class Layer():
    def __init__(self, num_neurons, num_previous_neurons, initial_weight_min, initial_weight_max):
        self.neurons = []
        self.num_neurons = num_neurons
        self.outputs = np.empty(num_neurons)
        self.weight_matrix = np.empty([num_neurons, num_previous_neurons])
        #initialize weights
        for i in range(num_neurons):
            neuron = Neuron(num_previous_neurons)
            for j in range(num_previous_neurons):
                neuron.weights[j] = random.uniform(initial_weight_min, initial_weight_max)
                #mat mul
                self.weight_matrix[i][j] = random.uniform(initial_weight_min, initial_weight_max)
            self.neurons.append(neuron)
'''

class NeuralNetwork():
    def __init__(self, num_inputs, layers, num_outputs, activation_function, final_function, batch_size):
        self.num_inputs = num_inputs
        self.input_layer = np.empty(0)
        self.num_layers = len(layers)
        self.layers = layers
        self.num_outputs = num_outputs
        self.outputs = np.empty(num_outputs)
        self.expected = np.empty(num_outputs)
        self.activation_function = activation_function
        self.final_function = final_function
        self.batch_size = batch_size
        self.batch_count = 0
        self.weights = []
        self.gradients = []

    def initialize_expected(self, expected):
        for neuron, expected_output in zip(self.layers[-1].neurons, expected):
            neuron.expected = expected_output

    def initialize_input_layer(self, inputs):
        self.input_layer = np.empty(0)
        for input in inputs:
            self.input_layer = np.append(self.input_layer, input)

    def forward_feed(self, inputs):
        self.initialize_input_layer(inputs)

        #for scope
        initial_layer_end_time = 0
        for i in range(self.num_layers):
            if i == 0:
                self.layers[i].outputs = np.matmul(self.layers[i].weight_matrix, inputs)
                '''
                for neuron in self.layers[i].neurons:
                    input = np.matmul(neuron.weights, self.input_layer)
                    neuron.input = input
                    neuron.output = self.activation_function(input)
                '''
                initial_layer_end_time = time.time()
                #print("time to initialize first layer:", initial_layer_end_time - initialize_inputs_end_time)
            else:
                self.layers[i].outputs = np.matmul(self.layers[i].weight_matrix, self.layers[i-1].outputs)
                if i != (self.num_layers-1):
                    vfunc = np.vectorize(self.activation_function)
                    self.layers[i].outputs = vfunc(self.layers[i].outputs)
                else:
                    #vfunc = np.vectorize(self.final_function)
                    #self.layers[i].outputs = vfunc(self.layers[i].outputs, self.layers[i].outputs)
                    outputs = self.final_function(self.layers[-1].outputs)
                    #for j in range(self.layers[-1].num_neurons):
                    self.layers[-1].outputs = outputs
                '''
                for j in range(self.layers[i].num_neurons):
                    input = 0
                    #previous_layer = self.layers[i-1]
                    #neuron = self.layers[i].neurons[j]
                    for k in range(self.layers[i-1].num_neurons):
                        input += self.layers[i].neurons[j].weights[k] * \
                            self.layers[i-1].neurons[k].output
                    self.layers[i].neurons[j].input = input
                    if i != (self.num_layers-1):
                        self.layers[i].neurons[j].output = self.activation_function(input)
                    # output layer
                    else:
                        outputs = self.final_function(self.layers[-1])
                        for j in range(self.layers[-1].num_neurons):
                            self.layers[-1].neurons[j].output = outputs[j]
                '''
                second_and_beyond_layer_end_time = time.time()
                #print("time to initialize second and beyond layer:", second_and_beyond_layer_end_time - initial_layer_end_time)
    
    #move left to right
    #after each layer, move left
    def calculate_gradients(self):
        for i in range(self.num_layers):
            for k in range(self.layers[i].num_neurons):
                neuron = self.layers[i].neurons[k]
                gradients = np.empty(0)
                Ij = neuron.input
                for j in range(neuron.num_weights):
                    #first hidden layer
                    if i == 0:
                        gradient = self.input_layer[j] * calculate_derivative_sigmoid(Ij) * self.calculate_EtoOj(neuron, i)
                        gradients = np.append(gradients, gradient)
                    else:
                        gradient = self.layers[i-1].neurons[j].output * calculate_derivative_sigmoid(Ij) * self.calculate_EtoOj(neuron, i)
                        gradients = np.append(gradients, gradient)
                #neuron.gradient_table = np.append([neuron.gradient_table], [gradients], axis = 0)
                        

    def calculate_EtoOj(self, neuron, layer_index):
        if layer_index != self.num_layers-1:
            value = 0
            #layer = self.layers[layer_index]
            #for neuron in layer.neurons:
            next_layer = self.layers[layer_index+1]
            for weight, next_neuron in zip(neuron.weights, next_layer.neurons):
                value_jk = weight * calculate_derivative_sigmoid(next_neuron.input) * self.calculate_EtoOj(next_neuron, layer_index+1)
                value += value_jk
            return value
        #output layer
        else:
            return -1 * (neuron.expected - neuron.output)


    # average the gradients across each weight
    # weight = weight - learning_rate * gradient
    def back_propagate(self, learning_rate):
        pass
        # print(dl_dw11)
        #weight = weight - learning_rate * dl_dw11

    # def update_weights(self, expected, learning_rate):
    def update_weights(self, final_outputs, expected_outputs, learning_rate):
        pass


if __name__ == "__main__":
    start_time = time.time()
    
    #train
    images = []
    labels = []
    print("take in input")
    images, labels = read_input(sys.argv[1], sys.argv[2])
    batch_size = 500
    num_training_rows = len(images)
    sample_size = 10000
    num_epochs = 20

    layers = []
    num_inputs = 28*28
    layer_1_num_neurons = 4
    num_outputs = 10
    initial_weight_min = -0.1
    initial_weight_max = 0.1

    print("time taken:", time.time() - start_time)

    print("initialize the neural network")
    layer_1 = Layer(layer_1_num_neurons, num_inputs, initial_weight_min, initial_weight_max)
    output_layer = Layer(num_outputs, layer_1_num_neurons, initial_weight_min, initial_weight_max)
    layers.append(layer_1)
    layers.append(output_layer)
    neuralNetwork = NeuralNetwork(
        num_inputs, layers, num_outputs, sigmoid_function, softmax_function, batch_size)
    
    print("time taken:", time.time() - start_time)

    # each set in the batch comes with an image and a label.
    # update weights after a batch
    #batch_num = 0
    for epoch_num in range(num_epochs):
        print("epoch", epoch_num, "started")
        #sample "sample_size" numbers
        print(range(num_training_rows), sample_size)
        sample_indices = random.sample(range(num_training_rows), sample_size)
        for trial_index in sample_indices:
            inputs = images[trial_index]
            label = int(labels[trial_index])
            expected = np.zeros(10)
            expected[label] = 1
            forward_feed_start_time = time.time()
            neuralNetwork.forward_feed(inputs)
            forward_feed_end_time = time.time()
            #print("forward feed time:", forward_feed_end_time - forward_feed_start_time)
            #time.sleep(1)
            neuralNetwork.initialize_expected(expected)
            calculate_gradients_start_time = time.time()
            neuralNetwork.calculate_gradients()
            calculate_gradients_end_time = time.time()
            #print("calculate gradients time:", calculate_gradients_end_time - calculate_gradients_start_time)
                #for weight, gradient in zip(neuron.weights, neuron.gradients):
                #    print("weight", weight, "gradient", gradient)
        print("epoch completed in ", time.time() - start_time)

    print("training time", time.time() - start_time)
    

    #test


    #ingest file one by one
    #faster trial process
    #30 / 200 epochs = 0.15 min per epoch
    
    #np.savetxt("foo.csv", a, delimiter=",")
