import matplotlib.pyplot as plt
import pickle

learning_rate = 0.1
rate = str(learning_rate)
file_name= "lists_"+rate+".pkl"
with open(file_name, 'rb') as file:
    loaded_lists = pickle.load(file)

training_losses, test_losses, train_accuracies, test_accuracies = loaded_lists

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot( training_losses, marker='o', linestyle='-')
plt.title(f'Iterations vs. Training Loss (learning rate = {learning_rate})')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
# plt.ylim(0.351, 0.353)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot( test_losses, marker='o', linestyle='-')
plt.title(f'Iterations vs. Testing Loss (learning rate = {learning_rate})')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(train_accuracies, marker='o', linestyle='-', label='Training Accuracy', color='blue',alpha=0.6)
plt.plot(test_accuracies, marker='o', linestyle='-', label='Testing Accuracy', color='red', alpha=0.6)
plt.title(f'Iterations vs. Training and Test Accuracy (learning rate = {learning_rate})')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
# plt.ylim(0, 1)  # Set the y-axis range
plt.legend()  # Show the legend with labels
plt.show() 