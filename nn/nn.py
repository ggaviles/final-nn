# Imports
import numpy as np
from typing import List, Dict, Tuple, Union

from matplotlib import pyplot as plt
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(self, nn_arch: List[Dict[str, Union[int, str]]],
                 lr: float, seed: int, batch_size: int, epochs: int, loss_function: str):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(self, W_curr: ArrayLike, b_curr: ArrayLike,
                        A_prev: ArrayLike, activation: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        # Set Z_curr to be weights matrix * activation matrix + biases
        Z_curr = np.add(np.dot(W_curr, A_prev), b_curr)

        # If 'relu' found in activation function name, apply ReLU function to Z_curr
        if "relu" in activation:
            A_curr = self._relu(Z_curr)

        # If 'sigmoid' found in activation function name, apply sigmoid function to Z_curr
        elif "sigmoid" in activation:
            A_curr = self._sigmoid(Z_curr)

        else:
        # If neither 'relu' or 'sigmoid' is mentioned in activation function name, raise ValueError
            raise NameError("Activation function name is not defined. "
                            "Choose either ReLU or sigmoid as activation function.")

        return A_curr, Z_curr  # Return a tuple of the activation matrix and linear transformed matrix

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        # Set A_curr to initial input matrix X
        A_curr = X

        # Make dictionary to store Z and A matrices from '_single_forward' pass
        # Set first key in dictionary to be input matrix X
        cache = {'A0': X}

        # Iterate through neural network
        for idx, layer in enumerate(self.arch):
            # Set A current to be the previous A matrix in anticipation of the next forward pass
            A_prev = A_curr

            # Set index of first layer to 1
            layer_idx = idx + 1

            # Extract corresponding values from param_dict
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            activation = layer['activation']  # Extract activation function string

            # Pass param_dict values to single forward pass method
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_prev, activation)

            # Update cache values
            cache['A' + str(layer_idx)] = A_curr
            cache['Z' + str(layer_idx)] = Z_curr

        return A_curr.T, cache

    def _single_backprop(self, W_curr: ArrayLike, b_curr: ArrayLike, Z_curr: ArrayLike, A_prev: ArrayLike,
                         dA_curr: ArrayLike, activation_curr: str) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        # If 'relu' found in current activation function name, calculate derivative of ReLU function to backpropagate
        if "relu" in activation_curr:
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)

        # If 'sigmoid' found in current activation function name, calculate derivative of sigmoid function to backpropagate
        elif "sigmoid" in activation_curr:
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)
        else:
            # If neither 'relu' or 'sigmoid' is mentioned in activation function name, raise ValueError
            raise NameError("Activation function name is not defined. "
                            "Choose either ReLU or sigmoid as activation function.")

        # Compute the gradients for current layer
        dW_curr = np.dot(dZ_curr, A_prev.T) / np.shape(A_prev)[1]
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / np.shape(A_prev)[1]
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        # Initialize grad_dict
        grad_dict = {}

        # Calculate dA_prev
        if 'binary cross entropy' in self._loss_func:
            dA_prev = self._binary_cross_entropy_backprop(y, y_hat)
        elif 'mean squared error' in self._loss_func:
            dA_prev = self._mean_squared_error_backprop(y, y_hat)
        else:
            raise NameError('Loss function name is not defined. '
                            'Loss function must be binary cross entropy or mean squared error.')

        # Iterate through neural network
        for idx, layer in reversed(list(enumerate(self.arch))):
            layer_idx = idx + 1  # Set index of first layer to 1
            activation = layer['activation']  # Extract activation function string
            dA_curr = dA_prev  # Set previous dA to curr dA

            if idx == 0:  # If at the first layer, A_prev will be equal to unactivated input matrix
                A_prev = cache['A0']
            else:
                A_prev = cache['A' + str(idx)]

            # Extract corresponding values from param_dict
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            Z_curr = cache['Z' + str(layer_idx)]

            # Pass param_dict values to single forward pass method
            dA_prev, dW_curr, db_curr = self._single_backprop(
                W_curr, b_curr, Z_curr, A_prev, dA_curr, activation
            )

            # Make dictionary to store gradient values
            grad_dict['dA' + str(layer_idx)] = dA_prev
            grad_dict['dW' + str(layer_idx)] = dW_curr
            grad_dict['db' + str(layer_idx)] = db_curr

        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        for index, layer in enumerate(self.arch):
            index += 1
            self._param_dict['W' + str(index)] -= self._lr * grad_dict['dW' + str(index)]
            self._param_dict['b' + str(index)] -= self._lr * grad_dict['db' + str(index)]

    def fit(self, X_train: ArrayLike, y_train: ArrayLike,
            X_val: ArrayLike, y_val: ArrayLike) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        # Initialize lists of training and validation losses
        per_epoch_loss_train = []
        per_epoch_loss_val = []

        # Iterate across each epoch
        for epoch in range(self._epochs):
            # Shuffle indices to randomly select batches
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)

            # Initialize list to keep track of different batches
            batches = []

            # Iterate through rows of X_train in increments of batch size
            for i in range(0, X_train.shape[0], self._batch_size):
                X_batch = X_train[i:i + self._batch_size]  # Use batch indices to select corresponding rows from X_train
                y_batch = y_train[i:i + self._batch_size]  # Use batch indices to select corresponding rows from y_train
                batches.append((X_batch, y_batch))  # Add the batch to the list of batches

            # Iterate through the X and y batches
            for X_batch, y_batch in batches:
                X_batch = X_batch.T  # Transpose each X batch
                y_batch = y_batch.T  # Transpose each X batch
                y_hat, cache = self.forward(X_batch)  # Forward pass on X batches
                grad_dict = self.backprop(y_batch, y_hat, cache)  # Backpropagate using y_batches
                self._update_params(grad_dict)  # Update parameters using values stored from backpropagation

            # Calculate predictions from training and validation sets
            y_hat_train = self.predict(X_train)
            y_hat_val = self.predict(X_val)

            # Calculate training and validation losses using user-defined loss function
            if "binary cross entropy" in self._loss_func:
                train_loss = self._binary_cross_entropy(y_train.T, y_hat_train.T)
                val_loss = self._binary_cross_entropy(y_val.T, y_hat_val.T)
            elif "mean squared error" in self._loss_func:
                train_loss = self._mean_squared_error(y_train.T, y_hat_train.T)
                val_loss = self._mean_squared_error(y_val.T, y_hat_val.T)
            else:
                raise NameError("Loss function name is not defined. "
                                "Choose either mean squared error or binary cross entropy as loss function.")

            # Append calculated loss to per epoch loss lists
            per_epoch_loss_train.append(train_loss)
            per_epoch_loss_val.append(val_loss)

        return per_epoch_loss_train, per_epoch_loss_val

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        y_hat, _ = self.forward(X.T)
        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return 1 / (1 + np.exp(-Z))

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        return dA * self._sigmoid(Z) * (1 - self._sigmoid(Z))

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return np.maximum(0, Z)

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        loss = -(1 / y.shape[1]) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return loss

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        dA = - (np.divide(y, y_hat.T) - np.divide((1 - y), (1 - y_hat.T)))
        return dA

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        return (1 / (2 * y.shape[1])) * np.sum(np.square(y_hat - y))

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        # Derivative of the MSE loss function
        return (2 / y.shape[1]) * (y_hat.T - y)