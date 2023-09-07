import numpy as np
import random
import csv
import matplotlib.pyplot as plt

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

    def backward(self):
        der_softmax_and_cost = self.input_for_next_layer[-1] - self.output_values
        backderivative= der_softmax_and_cost
        # new_weights= []
        # new_biases= []
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
            # new_biases.insert(0,self.biases[i]-learning_rate*backderivative)
            bias_derivative.insert(0,backderivative)
            
            # Adjust the weights
            adjust = np.dot(backderivative,self.input_for_next_layer[i].T)

            # new_weights.insert(0,self.weights[i] - learning_rate*adjust)
            weight_derivative.insert(0,adjust)

            backderivative = np.dot(self.weights[i].T,backderivative)
            

        # self.weights = new_weights
        # self.biases = new_biases
        return weight_derivative,bias_derivative

    # def update(self):

            
    
    def train(self,input_list,output_list,loss_function:LossFunctions,learning_rate:float,batch_size=128,epoch_size=100): 
        self.loss_function = loss_function
        losses =[]
        test_losses =[]
        test_accuracies =[]
        train_accuracies =[]
        i = 0
        with open("train_lost.txt", 'w') as output_file:
            
            input_sample_length = (len(output_list)//batch_size)
            for epoch in range(epoch_size):
                
                combined_lists = list(zip(input_list,output_list))
                random.shuffle(combined_lists)
                shuffled_list1, shuffled_list2 = zip(*combined_lists)
                epoch_loss = 0

                for i in range (input_sample_length):
                    batch_weight_derivative =[]
                    batch_bias_derivatives = []
                    batch_loss =0
                    for j in range(batch_size):
                        x = shuffled_list1[i*batch_size+j]
                        y = shuffled_list2[i*batch_size+j]
                        self.set_input_output(x,y)
                        self.foreward()

                        batch_loss += loss_function(y_true=self.output_values,y_pred=self.input_for_next_layer[-1])

                        weight_derivative,bias_derivative = self.backward()
                        batch_weight_derivative.append(weight_derivative)
                        batch_bias_derivatives.append(bias_derivative)
                    
                    batch_loss = batch_loss/batch_size
                    epoch_loss+=batch_loss
                    # losses.append(batch_loss)

                    loss_info = f"The loss of the epoch {epoch} batch {i} is:   {batch_loss}\n"
                    output_file.write(loss_info)

                    sum_weight_derivative = batch_weight_derivative[0]
                    sum_bias_derivative = batch_bias_derivatives[0]
                    for i in range(1,batch_size):
                        for j in range(self.size-1): # len(batch_weight_derivative[i])
                            sum_weight_derivative[j] = sum_weight_derivative[j]+batch_weight_derivative[i][j]
                            sum_bias_derivative[j] = sum_bias_derivative[j]+batch_bias_derivatives[i][j]

                    for k in range(self.size-1): # len(sum_weight_derivative)
                        sum_weight_derivative[k] = sum_weight_derivative[k]/batch_size
                        self.weights[k] = self.weights[k] - learning_rate*sum_weight_derivative[k]

                        sum_bias_derivative[k] = sum_bias_derivative[k]/batch_size
                        self.biases[k] = self.biases[k] - learning_rate*sum_bias_derivative[k]

                losses.append(epoch_loss/epoch_size)
                train_inputs, train_outputs = process(x='x_train.csv',y="y_train.csv")
                print(f"epoch {epoch+1} train", end=" ")
                train_loss,train_accuracy = neural.predict(train_inputs,train_outputs)
                train_accuracies.append(train_accuracy)

                test_inputs, test_outputs = process(x='x_test.csv', y="y_test.csv")
                print(f"epoch {epoch+1} test", end=" ")
                test_loss,test_accuracy = neural.predict(test_inputs,test_outputs)
                print()
                test_losses.append(sum(test_loss)/len(test_loss))
                test_accuracies.append(test_accuracy)
        return losses,test_losses,train_accuracies,test_accuracies
    
    def predict(self, input_list, output_list):
        losses=[]
        correctly_classified =0
        incorrectly_classified =0
        i=0
        with open("test_lost.txt", 'w') as output_file, open('test_classifications.csv', 'w') as file:
            for x, y in zip(input_list,output_list):
                self.set_input_output(x,y)
                self.foreward()
                i+=1
                
                loss = self.loss_function(y_true=self.output_values,y_pred=self.input_for_next_layer[-1])
                losses.append(loss)

                loss_info = f"The loss of the testing sample {i} is:   {loss}\n"
                output_file.write(loss_info)

                max_index = np.argmax(self.input_for_next_layer[-1]) # self.input_for_next_layer[-1] is same as softmax output

                if y[max_index]==1:
                    correctly_classified+=1
                else:
                    incorrectly_classified+=1

                file.write(f"{max_index}\n")
                
        # print(f"number of correctly classified samples: {correctly_classified}")
        # print(f"number of incorrectly classified samples: {incorrectly_classified}")
        accuracy = correctly_classified/(correctly_classified+incorrectly_classified)
        print(f"Accuracy: {accuracy}")

        return losses,accuracy

            

def process(x,y):
    one_hot_encoded_y = []
    list_x =[]
    with open(y, 'r') as y_file:
        csv_reader = csv.reader(y_file)
        for row in csv_reader:
            label = int(row[0])
            one_hot_label = [0] * 4  # Assuming 4 classes (0, 1, 2, 3)
            one_hot_label[label] = 1
            one_hot_encoded_y.append(one_hot_label)

    with open(x, 'r') as x_file:
        csv_reader_x = csv.reader(x_file)
        
        # Iterate through each row in the CSV file
        for x_row in csv_reader_x:
            x_data = [int(value) for value in x_row]
            # Check if the row contains exactly 14 integers
            if len(x_data) == 14:
                list_x.append(x_data)
            else:
                raise ValueError(f"Skipping row with incorrect number of values: {row}")
            
    return list_x , one_hot_encoded_y
    

train_inputs, train_outputs = process(x='x_train.csv',y='y_train.csv')

neural = NeuralNetwork([14,100,40,4])
neural.create()
neural.set_activations([ActivationFunctions.relu,ActivationFunctions.relu,ActivationFunctions.softmax])
learning_rate = 1
train_losses,test_losses,train_accuracies,test_acuracies= neural.train(input_list=train_inputs,output_list=train_outputs,loss_function=LossFunctions.categorical_crossentropy,
                            learning_rate=learning_rate,batch_size=64,epoch_size=200)

# test_inputs, test_outputs = process(x='x_test.csv', y="y_test.csv")
# neural.predict(test_inputs,test_outputs)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot( train_losses, marker='o', linestyle='-')
plt.title(f'Iterations vs. Training Loss (learning rate = {learning_rate})')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot( test_losses, marker='o', linestyle='-')
plt.title(f'Iterations vs. Test Loss (learning rate = {learning_rate})')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot( train_accuracies, marker='o', linestyle='-')
plt.title(f'Iterations vs. Training Accuracy (learning rate = {learning_rate})')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot( test_acuracies, marker='o', linestyle='-')
plt.title(f'Iterations vs. Test Accuracy (learning rate = {learning_rate})')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()