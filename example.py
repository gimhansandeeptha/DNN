import numpy as np

# Create a column vector
m1 = [[1],
               [2],
               [3]]
m3= np.array([[1,2,3]])
# Create a row vector
backderivative = np.array([[2.12],[5.4],[-1.5]])
print(backderivative)
print(backderivative[1][0])

input_for_next_layer=[]
input_for_next_layer.append(np.array([[1],[5.2],[4]]))
input_for_next_layer.append(np.array([[2.1],[3.5]]))

print(input_for_next_layer[0][1])