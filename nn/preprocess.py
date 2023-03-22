# Imports
import numpy as np
from random import choices
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # Initialize lists of positive and negatives sequences
    pos_seqs = []
    neg_seqs = []

    # Separate sequences into positive or negative lists depending on the presence of a corresponding label
    for seq, label in zip(seqs, labels):
        if label:
            pos_seqs.append(seq)
        else:
            neg_seqs.append(seq)

    # Calculate number of sequences in each class
    num_pos_seqs = len(pos_seqs)
    num_neg_seqs = len(neg_seqs)
    num_samples = min(num_pos_seqs, num_neg_seqs)

    # Randomly sample sequences with replacement
    pos_samples = choices(pos_seqs, k=num_samples)
    neg_samples = choices(neg_seqs, k=num_samples)

    # Combine the sampled sequences and their labels
    samples = list(pos_samples) + list(neg_samples)
    sampled_labels = [1] * num_samples + [0] * num_samples

    # Shuffle the sequences and labels
    indices = np.random.permutation(len(samples))
    sampled_seqs = [samples[i] for i in indices]
    sampled_labels = [sampled_labels[i] for i in indices]

    return sampled_seqs, sampled_labels

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    # Define one-hot encoding dictionary
    encoding_dict = {'A': [1, 0, 0, 0],
                     'T': [0, 1, 0, 0],
                     'C': [0, 0, 1, 0],
                     'G': [0, 0, 0, 1]}

    encoding_list = []  # Initialize an empty list to store the encodings

    # Iterate over each sequence in seq_arr
    for seqs in seq_arr:
        # Initialize an empty list to store the encoding of a sequence
        seq_encoding = []
        # Iterate over sequence nucleotides
        for nts in seqs:
            # Add encoding of each nucleotide to the encoding list
            seq_encoding += encoding_dict[nts]
        # Append the encoding to the list of encodings
        encoding_list.append(seq_encoding)

    # Convert the encodings list to a NumPy array and return it
    encoding_list = np.array(encoding_list)
    return encoding_list