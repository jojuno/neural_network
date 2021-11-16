import numpy
from io import StringIO

#take in input
#each input node represents a pixel
def read_input(image_file, label_file):
    images = numpy.genfromtxt(image_file, delimiter=",")
    labels = numpy.genfromtxt(label_file, delimiter="\n")
    return images, labels

if __name__ == "__main__":
    #feed forward network
        #take in a CSV file
        #feed it to two layers of hidden neurons
            #choose the number of neurons in each layer
        #feed it to the last layer
            #classify which number it is by taking the maximum of the neurons

    images = []
    labels = []
    images, labels = read_input("train_image1.csv", "train_label.csv")
    for image in images:
        inputs = image
    

