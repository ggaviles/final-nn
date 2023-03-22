# Imports
import numpy as np
from typing import List, Dict, Tuple, Union

from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from preprocess import one_hot_encode_seqs

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

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

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

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
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
        Z_curr = np.add((A_prev.dot(W_curr.T)), b_curr.T)

        if "relu" in activation:
            A_curr = self._relu(Z_curr)
        elif "sigmoid" in activation:
            A_curr = self._sigmoid(Z_curr)
        else:
            raise Exception("Choose either ReLu or sigmoid as activation function.")

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
        cache = {}
        cache['A0'] = X

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
        return A_curr, cache

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
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

        if "relu" in activation_curr:
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)

            # dW_curr = 1/m * dZ_curr . (A_prev).T
            dW_curr = np.dot(Z_curr.T, A_prev) / np.shape(A_prev)[1]

            # db_curr = 1/m * sum(dZ_curr)
            db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / np.shape(A_prev)[1]

            dA_prev = np.dot(dZ_curr, W_curr)

        elif "sigmoid" in activation_curr:
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)

            # dW_curr = 1/m * dZ_curr . (A_prev).T
            dW_curr = np.dot(Z_curr.T, A_prev) / np.shape(A_prev)[1]

            # db_curr = 1/m * sum(dZ_curr)
            db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / np.shape(A_prev)[1]

            dA_prev = np.dot(dZ_curr, W_curr)
        else:
            raise Exception("Choose either ReLu or sigmoid as activation function.")

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
            raise Exception('Loss function must be binary cross entropy or mean squared error.')

        # Iterate through neural network
        for idx, layer in reversed(list(enumerate(self.arch))):
            # Set index of first layer to 1
            layer_idx = idx + 1

            activation = layer['activation']  # Extract activation function string

            dA_curr = dA_prev

            if idx == 0:
                A_prev = cache['A0']
            else:
                A_prev = cache['A' + str(idx)]

            # Extract corresponding values from param_dict
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            Z_curr = cache['Z' + str(layer_idx)]

            # Pass param_dict values to single forward pass method
            dA_prev, dW_curr, db_curr = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation)

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
        for idx, layer in enumerate(self.arch):
            idx += 1
            W = self._param_dict['W' + str(idx)]
            dW = grad_dict['dW' + str(idx)]
            b = self._param_dict['b' + str(idx)]
            # db = grad_dict['db' + str(idx)]
            self._param_dict['W' + str(idx)] = W - (dW * self._lr)

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
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

        """# Define empty lists to store losses over training
        per_epoch_loss_train = []
        per_epoch_loss_val = []

        epochs = []

        # Repeat until convergence or maximum iterations reached
        for i in range(self._epochs):
            epochs.append(i+1)

            # Shuffling the training data for each epoch of training
            print('X_train is: ', X_train)
            print('y_train is: ', y_train)
            shuffle_arr = np.concatenate([X_train, np.expand_dims(y_train, 1)], axis=1)
            np.random.shuffle(shuffle_arr)
            X_train = shuffle_arr[:, :-1]
            y_train = shuffle_arr[:, -1].flatten()

            # Create batches
            num_batches = int(X_train.shape[0] / self._batch_size) + 1
            X_batch = np.array_split(X_train, num_batches)
            y_batch = np.array_split(y_train, num_batches)

            # Iterate through batches (one of these loops is one epoch of training)
            for X_train, y_train in zip(X_batch, y_batch):
                # Make prediction and calculate loss
                y_hat_train = self.predict(X_train)
                y_hat_val = self.predict(X_val)
                if 'binary cross entropy' in self._loss_func:
                    train_loss = self._binary_cross_entropy(y_train, y_hat_train)
                    per_epoch_loss_train.append(train_loss)

                    val_loss = self._binary_cross_entropy(y_val, y_hat_val)
                    per_epoch_loss_val.append(val_loss)

                elif 'mean squared error' in self._loss_func:
                    train_loss = self._mean_squared_error(y_train, y_hat_train)
                    per_epoch_loss_train.append(train_loss)

                    val_loss = self._mean_squared_error(y_val, y_hat_val)
                    per_epoch_loss_val.append(val_loss)

                else:
                    raise Exception('Choose loss function binary cross entropy or mean squared error')

                _, cache = self.forward(X_train)
                grad_dict = self.backprop(y_train, y_hat_train, cache)
                self._update_params(grad_dict)

        plt.plot(epochs, per_epoch_loss_train, label='Training Error')
        plt.plot(epochs, per_epoch_loss_val, label='Validation Error')
        plt.legend()
        plt.show()

        return per_epoch_loss_train, per_epoch_loss_val"""

        # Define empty lists to store losses over training
        per_epoch_loss_train = []
        per_epoch_loss_val = []
        epochs = []
        for i in range(self._epochs):
            epochs.append(i + 1)
            A_train, cache_t = self.forward(X_train)
            A_val, cache_v = self.forward(X_val)
            if self._loss_func == 'binary cross entropy':
                per_epoch_loss_train.append(self._binary_cross_entropy(y_train, A_train))
                per_epoch_loss_val.append(self._binary_cross_entropy(y_val, A_val))
            else:
                per_epoch_loss_train.append(self._mean_squared_error(y_train, A_train))
                per_epoch_loss_val.append(self._mean_squared_error(y_val, A_val))
            grad_dict = self.backprop(y_train, A_train, cache_t)
            self._update_params(grad_dict)

        plt.plot(epochs, per_epoch_loss_train, label='Training Error')
        plt.plot(epochs, per_epoch_loss_val, label='Validation Error')
        plt.legend()
        plt.show()

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
        """if X.shape[1] == self.num_feats:
            X = np.hstack([X, np.ones((X.shape[0], 1))])"""
        y_hat, cache = self.forward(X)
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
        nl_transform = 1 / (1 + np.exp(-Z))
        nl_transform = np.rint(nl_transform)
        nl_transform = np.asarray(nl_transform, dtype='int')
        return nl_transform

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
        sigmoid_output = self._sigmoid(Z)
        if np.shape(dA) != np.shape(sigmoid_output):
            dZ = dA.T * sigmoid_output * (1-sigmoid_output)
        else:
            dZ = dA * sigmoid_output * (1-sigmoid_output)

        """dZ = np.empty(len(dA))
        dA = np.array(dA, dtype=int)
        for i in range(len(dA) - 1):
            dZ_item = dA[i] * (1 - Z[i]) * Z[i]
            np.append(dZ, dZ_item)"""

        return dZ

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
        nl_transform = np.zeros((np.shape(Z)[0], np.shape(Z)[1]))
        for index1, i in enumerate(Z):
            for index2, j in enumerate(i):
                nl_transform[index1][index2] = max(0, j)
        return nl_transform

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
        # Derivative of ReLU and the chain rule
        deriv_relu = np.array(dA, copy=True)
        if np.shape(deriv_relu) != np.shape(Z):
            deriv_relu = np.transpose(deriv_relu)
            deriv_relu[Z <= 0] = 0
        else:
            deriv_relu[Z <= 0] = 0
        return deriv_relu

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
        y_hat = np.clip(y_hat, 1e-7, 1 - 1e-7)
        binary_cross_entropy = 0
        for index, i in enumerate(y_hat):
            y_curr = y[index][0]
            y_pred = y_hat[index][0]
            y_zero_loss = y_curr * np.log(y_pred)
            y_one_loss  = (1 - y_curr) * np.log(1 - y_pred)
            binary_cross_entropy += abs(y_zero_loss + y_one_loss)
        binary_cross_entropy = binary_cross_entropy / len(y)
        return binary_cross_entropy

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

        # Clip data to avoid division by 0
        y_hat = np.clip(y_hat, 1e-7, 1 - 1e-7)

        # Calculate gradient
        dA = -(y/y_hat - (1-y)/(1-y_hat))
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
        mean_squared_error = 0
        for index, y_pred in enumerate(y_hat):
            mean_squared_error += abs((y[0] - y_pred[0])**2)
        mean_squared_error = mean_squared_error/(2*len(y))
        return mean_squared_error

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
        # Derivative of the MSE loss function is 2/N * sum of i to N (y(i) - y_hat(i))
        dA = (2 / np.shape(y_hat)[1]) * np.transpose(y_hat - y)
        return dA
