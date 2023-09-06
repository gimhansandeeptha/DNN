import csv
import re
import numpy as np
class ActivationFunctions:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0)
    
weight_list =[]
# Open the CSV file for reading
with open('D:\Gimhan Sandeeptha\Gimhan\Semester 05\Deep Neural Networks\Assignment_1 Back Propagation\Task_1\\a\\w.csv', 'r') as csvfile:
    # Create a CSV reader
    csvreader = csv.reader(csvfile)
    weight_layer=[]
    activation_functions = []
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

bias_list =[]
# Open the CSV file for reading
with open('D:\Gimhan Sandeeptha\Gimhan\Semester 05\Deep Neural Networks\Assignment_1 Back Propagation\Task_1\\a\\b.csv', 'r') as csvfile:
    # Create a CSV reader
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        # Check if the row is not empty
        if row:
            data_values = [[float(value) for value in row[1:]]]
            bias_list.append(np.array(data_values).T)
            

print(bias_list)