import numpy as np

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