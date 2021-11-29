# neural_network
classifies numbers using layers of neurons

Input: handwritten digits converted into a 28x28 grid of pixels that represent the grayscale from 0 to 255.
Output: a prediction of the digit from 0 to 9.
Accuracy: +90%.

Configuration:
784 input neurons
3 hidden layers: 100 neurons, 75 neurons, and 50 neurons
10 output neurons: likelihood of the number being a digit from 0 to 9
Learning rate: time-scheduled; 0.2 at first, and changes to 0.002 at epoch 10
Biases: 1 on the input layer, and 1 at every hidden layer
Weights initialized with Xavier initialization; standard normal distribution multiplied by the square root of (4 / number of neurons of the next layer) (make it small so the sigmoid can handle it)

activation function in the hidden layers: sigmoid
function for probability distribution at the output layer: softmax

Libraries used: Numpy

Math: derivatives (chain rule), matrix multiplication, probability distribution, sigmoid function

Future plans: 
- create a program that takes in a photo, parses it into pixels (appropriate format for the input), and outputs the prediction in a user-friendly manner.
- apply the program to the alphabet.
