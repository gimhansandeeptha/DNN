import csv
import re
import numpy as np
import pandas as pd

class ActivationFunctions:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0)

class LossFunctions:

    @staticmethod
    def binary_crossentropy(y_true, y_pred):
        """
        Calculate the Binary Cross-Entropy Loss between binary true labels and predicted probabilities.
        """
        epsilon = 1e-15  # Small value to avoid division by zero
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip values to avoid log(0)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def categorical_crossentropy(y_true, y_pred):
        """
        Calculate the Categorical Cross-Entropy Loss between true one-hot encoded labels and predicted probabilities.
        """
        epsilon = 1e-15  # Small value to avoid division by zero
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip values to avoid log(0)
        return -np.sum(y_true * np.log(y_pred)) / len(y_true) # divided by number of outputs???????????
    

class NeuralNetwork:
    def __init__(self,neurons):
        self.neurons = neurons
        self.size = len(neurons)
        self.weights = []         # this is a list containing numpy arrays for each weight layer
        self.biases = []          # this is a list containig numpy arrys for each bias layer
        self.activations = []
        self.input_values = None  # after set_input_output method this is a two dimentional numpy array has only one inner list (as row vector)
        self.output_values = None # after set_input_output method this is a two dimentional numpy array has only one inner list (as row vector)
        self.loss_function = None
        self.input_for_next_layer = None   # input_for next layer's first row has the user inputs, other layers contain the output of the previous layers

    
    # get the input vector to the input_values
    def set_input_output(self, inputs, outputs):
        if isinstance(inputs, list) and isinstance(outputs, list):
            if all(isinstance(item, (int, float, str)) for item in inputs) and all(isinstance(item, (int, float, str)) for item in outputs):
                if self.neurons[0] == len(inputs) and self.neurons[-1] == len(outputs):
                    self.input_values = np.array([inputs]).T
                    self.output_values = np.array([outputs]).T
                else:
                    raise ValueError("The input or output is not compatible with the number of neurons in the first or last layer")
            else:
                raise ValueError("The input or output list is not one-dimensional.")
        else:
            raise TypeError("Input or output is not a list.")


    # this function has to be provided with the activation functions in the ActivationFunction class as a list.
    # each value in the activations represent the activation function in the each layer 
    def set_activations(self,activations):
        all_valid = all(hasattr(ActivationFunctions, func.__name__) for func in activations)
        if all_valid and len(activations) == self.size-1:
            self.activations = activations
        else:                    
            raise ValueError("Please enter the correct activation functions for the each layer ")
    
    # one woreword pass through the neural network
    def foreward(self):
        self.input_for_next_layer=[]
        output_vector = self.input_values
        self.input_for_next_layer.append(output_vector)
        for i in range(0,self.size-1):
            weighted_sum = np.dot(self.weights[i],output_vector)+ self.biases[i]
            output_vector = self.activations[i](weighted_sum)
            self.input_for_next_layer.append(output_vector)

    def backward(self,learning_rate):
        der_softmax_and_cost = self.input_for_next_layer[-1] - self.output_values
        backderivative= der_softmax_and_cost
        new_weights= []
        new_biases= []
        weight_derivative =[]
        bias_derivative =[]
        # network has to have at least one hidden layer
        for i in range(len(self.input_for_next_layer)-2,-1,-1):

            # get the derivative (1 or 0) for ReLU
            if self.activations[i] is ActivationFunctions.relu:
                for j in range(len(self.input_for_next_layer[i+1])):
                    if self.input_for_next_layer[i+1][j][0] <= 0:
                        backderivative[j][0] =0

            # Adjust the bias
            new_biases.insert(0,self.biases[i]-learning_rate*backderivative)
            
            # Adjust the weights
            adjust = np.dot(backderivative,self.input_for_next_layer[i].T)

            weight_derivative.insert(0,adjust)
            bias_derivative.insert(0,backderivative)

            new_weights.insert(0,self.weights[i] - learning_rate*adjust)
            backderivative = np.dot(self.weights[i].T,backderivative)
            

        self.weights = new_weights
        self.biases = new_biases
        return weight_derivative, bias_derivative
            
def process(input_weight_file,input_bias_file):
    weight_list =[]
    activation_functions = []
    # Open the CSV file for reading
    with open(input_weight_file, 'r') as csvfile:
        # Create a CSV reader
        csvreader = csv.reader(csvfile)
        weight_layer=[]
        # Iterate through each line in the CSV file
        i=0
        for row in csvreader:
            # Check if the row is not empty
            if row:
                first_element = row[0]
                pattern = r'layer(\d+) to layer(\d+)'
                
                match = re.search(pattern, first_element)
                
                if match:
                    layer1_number = int(match.group(1))
                    # layer2_number = int(match.group(2))

                    data_values = [float(value) for value in row[1:]]
                    if i == layer1_number:
                        weight_layer.append(data_values)
                        
                    else:
                        weight_list.append(np.array(weight_layer).T)
                        activation_functions.append(ActivationFunctions.relu)
                        weight_layer =[]
                        weight_layer.append(data_values)
                        i+=1   
                else:
                    raise ValueError("No match found.")

        weight_list.append(np.array(weight_layer).T)
        activation_functions.append(ActivationFunctions.softmax)
    # ---------------------------------------------------------------------------------

    bias_list =[]
    num_of_neurons = [14]
    # Open the CSV file for reading
    with open(input_bias_file, 'r') as csvfile:
        # Create a CSV reader
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            # Check if the row is not empty
            if row:
                data_values = [[float(value) for value in row[1:]]]
                num_of_neurons.append(len(data_values[0]))
                bias_list.append(np.array(data_values).T)

    return weight_list, bias_list, num_of_neurons,activation_functions
    
# give the path to the input weight file
input_weight_file = 'D:\Gimhan Sandeeptha\Gimhan\Semester 05\Deep Neural Networks\Assignment_1 Back Propagation\Task_1\\a\\w.csv' 

# give the path to the input bias file
input_bias_file = 'D:\Gimhan Sandeeptha\Gimhan\Semester 05\Deep Neural Networks\Assignment_1 Back Propagation\Task_1\\a\\b.csv'

# give the path to the output file to save weight derivatives
output_weight_file = 'D:\Gimhan Sandeeptha\Gimhan\Semester 05\Deep Neural Networks\Assignment_1 Back Propagation\csv_files\dw.csv'

# give the path to the output file to save bias derivatives
output_bias_file = 'D:\Gimhan Sandeeptha\Gimhan\Semester 05\Deep Neural Networks\Assignment_1 Back Propagation\csv_files\db.csv'

weight_list, bias_list , num_of_neurons, activation_functions = process(input_weight_file,input_bias_file)
neural = NeuralNetwork(num_of_neurons)
neural.set_input_output(inputs=[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1],outputs=[0,0,0,1])
neural.biases = bias_list
neural.weights= weight_list
neural.set_activations(activation_functions)
neural.foreward()
weight_derivative, bias_derivative = neural.backward(learning_rate=0.1)


with open(output_weight_file, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    for vec in weight_derivative:
        vec= vec.T
        for row in vec:
            csv_writer.writerow(row)

with open(output_bias_file, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    for vec in bias_derivative:
        vec= vec.T
        for row in vec:
            csv_writer.writerow(row)
