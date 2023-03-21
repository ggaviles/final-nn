# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
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
        nn_arch: List[Dict[str, Union(int, str)]],
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
        # Make a Z_curr matrix with the same dimensions as the multiplied matrices
        Z_curr = np.zeros(W_curr.dot(A_prev).shape)

        # Make a A_curr activation matrix with the same dimensions as the linear transformed matrix Z_curr
        A_curr = np.zeros(Z_curr.shape)

        # Set Z_curr to be weights matrix * activation matrix + biases
        Z_curr = W_curr.dot(A_prev) + b_curr

        # Update current activation matrix by applying a user-defined activation function to the Z_curr matrix
        #A_curr = globals()[activation](Z_curr)  # May have to put activation functions in global scope

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
        # Iterate through neural network
        for idx, layer in enumerate(self.arch):
            # Set index of first layer to 1
            layer_idx = idx + 1

            # Extract corresponding values from param_dict
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            A_prev = X  # Define first activation layer to be equal to X
            activation = layer['activation']  # Extract activation function string

            # Pass param_dict values to single forward pass method
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_prev, activation)

            # Make dictionary to store Z and A matrices from '_single_forward' pass
            cache = {}
            cache['A_curr' + str(layer_idx)] = A_curr
            cache['Z_curr' + str(layer_idx)] = Z_curr

            # Set A current to be the previous A matrix in anticipation of the next forward pass
            A_prev = A_curr

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
            dZ_curr = self._relu_backprop(A_prev, Z_curr)
            # dA_prev = (W_curr).T . dZ_curr
            dA_prev = np.dot(W_curr.T, dZ_curr)

            # dW_curr = 1/m * dZ_curr . (A_prev).T
            dW_curr = 1/A_prev.shape[1] * np.dot(dZ_curr, A_prev.T)

            # db_curr = 1/m * sum(dZ_curr)
            db_curr = 1/A_prev.shape[1] * np.sum(dZ_curr, axis=1)

        elif "sigmoid" in activation_curr:
            dZ_curr = self._sigmoid_backprop(A_prev, Z_curr)
            # dA_prev = (W_curr).T . dZ_curr
            dA_prev = np.dot(W_curr.T, dZ_curr)

            # dW_curr = 1/m * dZ_curr . (A_prev).T
            dW_curr = 1 / A_prev.shape[1] * np.dot(dZ_curr, A_prev.T)

            # db_curr = 1/m * sum(dZ_curr)
            db_curr = 1 / A_prev.shape[1] * np.sum(dZ_curr, axis=1)
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

        # Iterate through neural network
        for idx, layer in reversed(list(enumerate(self.arch))):
            # Set index of first layer to 1
            layer_idx = idx + 1

            # Extract corresponding values from param_dict
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            Z_curr = cache['Z_curr' + str(layer_idx)]
            A_curr = cache['A_curr' + str(layer_idx)]
            activation = layer['activation']  # Extract activation function string

            # Calculate dA_curr
            dA_curr = np.dot(W_curr.T, dA_curr) * self._sigmoid_backprop(A_curr, Z_curr)

            # Pass param_dict values to single forward pass method
            dA_prev, dW_curr, db_curr = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation)

            # Make dictionary to store gradient values
            grad_dict = {}
            grad_dict['dA_prev' + str(layer_idx)] = dA_prev
            grad_dict['dW_curr' + str(layer_idx)] = dW_curr
            grad_dict['db_curr' + str(layer_idx)] = db_curr

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
            layer_idx = idx + 1
            self._param_dict['W' + str(layer_idx)] \
                = self._param_dict['W' + str(layer_idx)] + self._lr * grad_dict['dW_curr' + str(layer_idx)]
            self._param_dict['b' + str(layer_idx)] \
                = self._param_dict['b' + str(layer_idx)] + self._lr * grad_dict['db_curr' + str(layer_idx)]

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
        # Define empty lists to store losses over training
        per_epoch_loss_train = []
        per_epoch_loss_val = []

        # Padding data with vector of ones for bias term
        X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
        X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])

        # Defining intitial values for while loop
        epoch_num = 1

        # Repeat until convergence or maximum iterations reached
        while epoch_num < self._epochs:

            # Shuffling the training data for each epoch of training
            shuffle_arr = np.concatenate([X_train, np.expand_dims(y_train, 1)], axis=1)
            np.random.shuffle(shuffle_arr)
            X_train = shuffle_arr[:, :-1]
            y_train = shuffle_arr[:, -1].flatten()

            # Create batches
            num_batches = int(X_train.shape[0] / self._batch_size) + 1
            X_batch = np.array_split(X_train, self._batch_size)
            y_batch = np.array_split(y_train, self._batch_size)

            # Create list to save the parameter update sizes for each batch
            update_sizes = []

            # Iterate through batches (one of these loops is one epoch of training)
            for X_train, y_train in zip(X_batch, y_batch):
                # Make prediction and calculate loss
                y_pred = self.predict(X_train)
                if 'binary_cross_entropy' in self._loss_func:
                    train_loss = self._binary_cross_entropy(y_train, y_pred)
                elif 'mean_squared_error' in self._loss_func:
                    train_loss = self._mean_squared_error(y_train, y_pred)
                else:
                    raise Exception('Choose loss function binary_cross_entropy or mean_squared_error')
                per_epoch_loss_train.append(train_loss)

                # Update weights

                """! Might be able to encapsulate all of this in backprop and update params"""
                prev_W = self.W
                if 'binary_cross_entropy' in self._loss_func:
                    grad = self._binary_cross_entropy_backprop(y_train, y_pred)
                elif 'mean_squared_error' in self._loss_func:
                    grad = self._mean_squared_error_backprop(y_train, y_pred)
                else:
                    raise Exception('Choose loss function binary_cross_entropy or mean_squared_error')
                new_W = prev_W - self.lr * grad
                self.W = new_W

                # Update weights
                # Maybe use update params here


                # Save parameter update size
                update_sizes.append(np.abs(new_W - prev_W))

                # Compute validation loss
                if 'binary_cross_entropy' in self._loss_func:
                    val_loss = self._binary_cross_entropy(y_val, self.predict(X_val))
                elif 'mean_squared_error' in self._loss_func:
                    val_loss = self._mean_squared_error(y_val, self.predict(X_val))
                else:
                    raise Exception('Choose loss function binary_cross_entropy or mean_squared_error')

                # Add validation loss to list
                per_epoch_loss_val.append(val_loss)

            # Define step size as the average parameter update over the past epoch
            prev_update_size = np.mean(np.array(update_sizes))

            # Update iteration
            epoch_num += 1

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
        A_curr, cache = self.forward(X)
        y_hat = A_curr
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
        dZ = dA * (1 - Z) * Z

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
        return np.maximum(0, Z)  # If greater than 0, return Z; If less than 0, return 0

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
        dZ = dA * (1. if Z > 0 else 0.)
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
        y_hat = np.clip(y_hat, 1e-9, 1 - 1e-9)

        # Calculate the separable parts of the loss function
        y_zero_loss = y * np.log(y_hat)
        y_one_loss = (1 - y) * np.log(1 - y_hat)
        return -np.mean(y_zero_loss + y_one_loss)

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
        num = y_hat.shape[0]

        # Clip data to avoid division by 0
        clipped_y_hat_values = np.clip(y_hat, 1e-7, 1 - 1e-7)

        # Calculate gradient
        dA = -(y / clipped_y_hat_values -
               (1 - y) / (1 - clipped_y_hat_values) / )

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
        d1 = y - y_hat
        mse = (1/int(len(y))*d1.dot(d1))
        return np.mean(np.square(y - y_hat))

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
        return 2 * np.mean(y - y_hat)