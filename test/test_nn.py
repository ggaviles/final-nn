import numpy as np
import pytest

import nn.nn
import nn.preprocess

nn_arch = [{'input_dim': 5, 'output_dim': 10, 'activation': 'relu'},
           {'input_dim': 10, 'output_dim': 2, 'activation': 'sigmoid'}]

neural_net = nn.nn.NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=8, epochs=100, loss_function="binary cross entropy")

def test_single_forward():  #
    X = np.array([[]])
    W_curr = neural_net._param_dict['W1']
    b_curr = neural_net._param_dict['b1']
    activation = "relu"
    A_curr, Z_curr = neural_net(X, W_curr, b_curr, activation)

    assert A_curr.shape == ()

def test_forward():  #
    # Test the forward method to ensure it computes the correct output shape
    X = np.array([[1, 2], [3, 4]])
    y_hat, cache = nn.forward(X)
    assert y_hat.shape == (2, 1)  # Expected output shape

def test_single_backprop():  #
    # Test the _single_backprop method to ensure it computes the correct gradient shapes
    dA_curr = np.array([[1, 2], [3, 4], [5, 6]])
    W_curr = nn._param_dict["W1"]
    Z_curr = np.array([[1, 2], [3, 4], [5, 6]])
    A_prev = np.array([[1, 2], [3, 4]])
    activation_curr = "relu"

    dA_prev, dW_curr, db_curr = nn._single_backprop(dA_curr, W_curr, Z_curr, A_prev, activation_curr)
    assert dA_prev.shape == (2, 2)  # Expected shape for the gradient dA
    assert dW_curr.shape == (3, 2)  # Expected shape for the gradient dW
    assert db_curr.shape == (3, 1)  # Expected shape for the gradient db

def test_predict():  #
    # Test the predict method to ensure it computes the correct output shape
    X = np.array([[1, 2], [3, 4]])
    y_hat = nn.predict(X)
    assert y_hat.shape == (2, 1)  # Expected output shape

def test_binary_cross_entropy():  #
    # Test the _binary_cross_entropy method to ensure it computes the correct loss value
    y = np.array([[1, 0], [0, 1]])
    y_hat = np.array([[0.7, 0.2], [0.3, 0.8]])
    loss = nn._binary_cross_entropy(y, y_hat)
    assert np.isclose(loss, 0.5798184952529422)  # Expected loss value

def test_binary_cross_entropy_backprop():  #
    # Test the _binary_cross_entropy_backprop method to ensure it computes the correct gradient shape
    y = np.array([[1, 0], [0, 1]])
    y_hat = np.array([[0.7, 0.2], [0.3, 0.8]])
    dA = nn._binary_cross_entropy_backprop(y, y_hat)
    assert dA.shape == (2, 2)  # Expected shape for the gradient dA

def test_mean_squared_error():  #
    # Test the _mean_squared_error method to ensure it computes the correct loss value
    y = np.array([[1, 0], [0, 1]])
    y_hat = np.array([[0.7, 0.2], [0.3, 0.8]])
    loss = nn._mean_squared_error(y, y_hat)
    expected_loss = 0.065  # Expected loss value
    assert np.isclose(loss, expected_loss)  # Check if the computed loss is close to the expected loss

def test_mean_squared_error_backprop():  #
    # Test the _mean_squared_error_backprop method to ensure it computes the correct gradient values
    y = np.array([[1, 0], [0, 1]])
    y_hat = np.array([[0.7, 0.2], [0.3, 0.8]])
    dA = nn._mean_squared_error_backprop(y, y_hat)
    expected_dA = np.array([[-0.3, 0.3], [0.2, -0.2]])  # Expected gradient values
    assert np.allclose(dA, expected_dA)  # Check if the computed gradient is close to the expected gradient

def test_sample_seqs():  #
    # Define an array of sample sequences from rap1-lieb positives dataset
    seqs = ['GAATCCGTACATTTAGA', 'CCACCCGTACACCTCCC', 'GCACCCGCGCCTTCCTC',
            'AAACCCGGACATTCCAT', 'ACACCCACACCCCTCAT', 'TGACCCATACATTTCCT',
            'GCATCCGTGCCTCCCAC', 'TAACCCATACACCTCAT', 'ACACCCATACAAACCCA']
    labels = [True, False, True, False, False, True, True, False, True]

    samples, sampled_labels = nn.preprocess.sample_seqs(seqs, labels)

    # Check that number of output sampled labels and sampled sequences are the same
    assert len(samples) == len(sampled_labels)

    # Check if the number of True and False labels in the output is the same
    assert sampled_labels.count(True) == sampled_labels.count(False)

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