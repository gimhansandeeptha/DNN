import pandas as pd
import numpy as np 
import math 

weight_file_path = "D:\\Gimhan Sandeeptha\\Gimhan\\Semester 05\\Deep Neural Networks\\Assignment_1 Back Propagation\\Task_1\\a\\w.csv"
bias_file_path = "D:\\Gimhan Sandeeptha\\Gimhan\\Semester 05\\Deep Neural Networks\\Assignment_1 Back Propagation\\Task_1\\a\\b.csv"

weight = pd.read_csv(weight_file_path,header=None,usecols=range(1, 101))
bias = pd.read_csv(bias_file_path,header=None,usecols=range(1, 2))

print(weight)

weight_arr = weight.values
print(weight_arr)


def defferentiate():
    pass