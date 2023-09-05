import numpy as np
import random

class ActivationFunctions:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0)

class LossFunctions:
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """
        Calculate the Mean Squared Error (MSE) loss between the true values and predicted values.
        """
        return np.mean((y_true - y_pred) ** 2)

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

    
    # create weight and bias for i +1 th layer by randomly assingned numbers between 0 and 1.
    # weight vector is two dimntional  vector. bias vector is one dimentional vector. 
    def create(self):  
        for i in range (self.size-1):
            weight = np.array([[random.random() for _ in range(self.neurons[i])] for _ in range(self.neurons[i+1])])
            bias = np.array([[random.random() for _ in range(self.neurons[i+1])]])
            self.weights.append(weight)
            self.biases.append(bias.T)

    
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
        # print(self.input_for_next_layer)

    def backward(self,learning_rate):
        der_softmax_and_cost = self.input_for_next_layer[-1] - self.output_values
        backderivative= der_softmax_and_cost
        new_weights= []
        new_biases= []

        # network has to have at least one hidden layer
        for i in range(len(self.input_for_next_layer)-2,-1,-1):
            # Adjust the bias
            # print(f"Input for next layer \n{self.input_for_next_layer}\n")
            # print(i)
            # print(self.biases[i])
            new_biases.insert(0,self.biases[i]-learning_rate*backderivative)
            
            # Adjust the weights
            adjust = np.dot(backderivative,self.input_for_next_layer[i].T)
            new_weights.insert(0,self.weights[i] - learning_rate*adjust)

            backderivative = np.dot(self.weights[i].T,backderivative)
            # print(f"Backderivative Before \n{backderivative}\n")
            if self.activations[i] is ActivationFunctions.relu:
                for j in range(len(self.input_for_next_layer[i])):
                    if self.input_for_next_layer[i][j][0] <= 0:
                        backderivative[j][0] =0
                # print(f"Backderivative After \n{backderivative}\n")

        # print(f"Old weights \n{self.weights} \n")
        self.weights = new_weights
        # print(f"new weights \n{self.weights} \n")

        # print(f"Old biases \n{self.biases} \n")
        self.biases = new_biases
        # print(f"new biases \n{self.biases} \n")
            
    
    def train(self,loss_function:LossFunctions,learning_rate:float): 
        self.loss_function = loss_function
        i = 0
        while True:
            self.foreward()
            i+=1
            # print(f"weights after {i} th foreword pass  \n{self.weights} \n")
            # print(f"biases after {i} th foreword pass  \n{self.biases} \n")
            
            loss = loss_function(y_true=self.output_values,y_pred=self.input_for_next_layer[-1])
            print(f"The loss of the iteration {i} is: \n{loss}\n")

            if loss<=0.01:
                break

            self.backward(learning_rate)
            # print(f"weights after {i} th backword pass  \n{self.weights} \n")
            # print(f"biases after {i} th backword pass  \n{self.biases} \n")
        

neural = NeuralNetwork([2,3,2])
neural.create()
neural.set_input_output(inputs=[1,2],outputs=[0,1])
neural.set_activations([ActivationFunctions.relu,ActivationFunctions.softmax])
neural.train(loss_function=LossFunctions.categorical_crossentropy,learning_rate=0.1)
# print()

