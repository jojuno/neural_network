'''
learning rate between 0.1 and 1 with increments of 0.05
np.sqrt(1 / ...) standard. try with 4 if anything
randomize biases between 0 and 1
do this 5-6 times
get the average of the best performance


optimize other parameters first (remove comments, unnecessary code, reduce calculations)
print the configurations
500 nodes, 1 layer, 0.7 learning rate
128 - 200 (increments of 4) nodes first layer, 50 - 130 (increments of 3 or 4) nodes second layer
'''

learning_rate = range(0.1, 1, 0.05)
bias_0 = math.