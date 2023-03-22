import numpy as np
import pytest

import nn.nn
import nn.preprocess

nn_arch = [{'input_dim': 5, 'output_dim': 10, 'activation': 'relu'},
           {'input_dim': 10, 'output_dim': 2, 'activation': 'sigmoid'}]

neural_net = nn.nn.NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=8, epochs=100, loss_function="binary cross entropy")

def test_single_forward():
    # Define an input matrix X
    X = np.array([[5, 6], [7, 8], [9, 10], [11, 12], [13, 14]])
    W_curr = neural_net._param_dict['W1']  # Extract W_curr value from param_dict
    b_curr = neural_net._param_dict['b1']  # Extract b_curr value from param_dict

    # Run a forward pass through a single layer using the activation function ReLU
    A_curr, Z_curr = neural_net._single_forward(W_curr, b_curr, X, "relu")

    # Check that activated matrix has the right shape
    assert A_curr.shape == (10, 2)

def test_forward():
    # Test forward method to ensure it computes the correct y_hat output shape
    X = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [6, 7, 8, 9, 10]])

    # Calculate predictions after a full forward pass through the neural network
    y_hat, _ = neural_net.forward(X)

    # Confirm that matrix of predictions has the right output shape
    assert y_hat.shape == (5, 2)

def test_single_backprop():
    # Test _single_backprop method to ensure it computes the correct gradient shapes
    dA_curr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 2, 3, 4, 5],
                        [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 2, 3, 4, 5]])
    W_curr = neural_net._param_dict["W1"]
    b_curr = neural_net._param_dict["b1"]
    Z_curr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 2, 3, 4, 5],
                        [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 2, 3, 4, 5]])
    A_prev = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

    dA_prev, dW_curr, db_curr = neural_net._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, "relu")
    assert dA_prev.shape == (5, 5)  # Assert that the calculated gradient dA has the expected shape
    assert dW_curr.shape == (10, 2)  # Assert that the calculated gradient dW has the expected shape
    assert db_curr.shape == (10, 1)  # Assert that the calculated gradient db has the expected shape

def test_predict():
    # Test predict method to ensure it computes the correct y_hat output shape
    X = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

    # Use predict method to call the forward method and complete a forward pass through the neural network
    y_hat = neural_net.predict(X)

    # Check that matrix of predicted values has expected output shape
    assert y_hat.shape == (4, 2)

def test_binary_cross_entropy():
    # Test _binary_cross_entropy method to ensure it computes the correct loss value

    # Define ground truth y and prediction values y_hat
    y = np.array([[0., 0., 1., 0., 1., 1., 1., 1., 1., 1.]])
    y_hat = np.array([[0.19, 0.33, 0.47, 0.7, 0.74, 0.81, 0.86, 0.94, 0.97, 0.99]])

    # Calculate loss by binary cross entropy method
    loss = neural_net._binary_cross_entropy(y, y_hat)

    # Check that calculated binary cross entropy loss value is close to the expected loss value
    assert np.isclose(loss, 0.3329, rtol=1e-2)

def test_binary_cross_entropy_backprop():
    # Test _binary_cross_entropy_backprop method to ensure it computes the gradient dA with the correct shape

    # Define ground truth y and prediction values y_hat
    y = np.array([[0., 0., 1., 0., 1., 1., 1., 1., 1., 1.]])
    y_hat = np.array([[0.19, 0.33, 0.47, 0.7, 0.74, 0.81, 0.86, 0.94, 0.97, 0.99]])

    # Calculate gradient dA by calling _binary_cross_entropy_backprop method
    dA = neural_net._binary_cross_entropy_backprop(y, y_hat)

    # Confirm that gradient dA has the expected shape
    assert dA.shape == (10, 10)

def test_mean_squared_error():
    # Test the _mean_squared_error method to ensure it computes the correct loss value

    # Define ground truth y and predicted values y_hat
    y = np.array([[0., 0., 1., 0., 1., 1., 1., 1., 1., 1.]])
    y_hat = np.array([[0.19, 0.33, 0.47, 0.7, 0.74, 0.81, 0.86, 0.94, 0.97, 0.99]])

    # Calculate loss using _mean_squared_error method
    loss = neural_net._mean_squared_error(y, y_hat)

    expected_loss = 0.052  # Expected loss value

    # Check that the computed loss is close to the expected loss
    assert np.isclose(loss, expected_loss, rtol=1e-02)

def test_mean_squared_error_backprop():
    # Test _mean_squared_error_backprop method to confirm that it computes the correct gradient values

    # Define ground truth y and predicted values y_hat
    y = np.array([[0., 0., 1., 0., 1., 1., 1., 1., 1., 1.]])
    y_hat = np.array([[0.19, 0.33, 0.47, 0.7, 0.74, 0.81, 0.86, 0.94, 0.97, 0.99]])

    # Calculate gradient dA by calling _mean_squared_error_backprop method
    dA = neural_net._mean_squared_error_backprop(y, y_hat)

    # Define expected gradient values for first row in
    expected_dA_0 = np.array([[ 0.038, 0.038, -0.162, 0.038, -0.162, -0.162, -0.162, -0.162, -0.162, -0.162]])
    assert np.allclose(dA[0], expected_dA_0)  # Check if the computed gradient is close to the expected gradient

def test_sample_seqs():
    # Define an array of sample sequences and labels from rap1-lieb positives dataset
    seqs = ['GAATCCGTACATTTAGA', 'CCACCCGTACACCTCCC', 'GCACCCGCGCCTTCCTC',
            'AAACCCGGACATTCCAT', 'ACACCCACACCCCTCAT', 'TGACCCATACATTTCCT',
            'GCATCCGTGCCTCCCAC', 'TAACCCATACACCTCAT', 'ACACCCATACAAACCCA']
    labels = [True, False, True, False, False, True, True, False, True]

    samples, sampled_labels = nn.preprocess.sample_seqs(seqs, labels)

    # Check that number of output sampled labels and sampled sequences are the same
    assert len(samples) == len(sampled_labels)

    # Check that the number of True and False labels in the output is the same
    assert sampled_labels.count(False) == sampled_labels.count(True)

def test_one_hot_encode_seqs():
    # Define an array of sample sequences from rap1-lieb positives dataset
    seqs = ['GAATCCGTACATTTAGA', 'CCACCCGTACACCTCCC', 'GCACCCGCGCCTTCCTC',
            'AAACCCGGACATTCCAT', 'ACACCCACACCCCTCAT', 'TGACCCATACATTTCCT',
            'GCATCCGTGCCTCCCAC', 'TAACCCATACACCTCAT', 'ACACCCATACAAACCCA']

    # Return the one-hot encodings of the sequences above
    encodings = nn.preprocess.one_hot_encode_seqs(seqs)

    # Check output shape: num of rows should be length of seq array,
    # num of cols should be len of seqs (17) * encoding for each nt (4)
    assert encodings.shape == (len(seqs), 17 * 4)

    # Check if number of rows in the output is equal to number of sample sequences
    assert encodings.shape[0] == len(seqs)

    # Check random encoding
    AGA_encoding = np.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0])

    # Return the one-hot encodings of the sequence above
    test_encoding = nn.preprocess.one_hot_encode_seqs('AGA')

    # Flatten test_encoding to one-dimensional array
    test_encoding_1D = np.array(test_encoding).ravel()

    assert AGA_encoding.all() == test_encoding_1D.all()