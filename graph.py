import numpy as np
import scipy.sparse as sp
import pandas as pd

def build_adjacency_matrix(entity_pairs, num_entities):
    data = np.ones(len(entity_pairs))
    rows = [pair[0] for pair in entity_pairs]
    cols = [pair[1] for pair in entity_pairs]
    adj = sp.coo_matrix((data, (rows, cols)), shape=(num_entities, num_entities))
    return adj

def build_feature_matrix(num_entities):
    features = np.eye(num_entities)
    return features

def build_label_vector(labels_raw, classification=True):
    if classification:
        # Assuming binary classification based on a threshold
        threshold = 0.5  # Set the threshold for converting probabilities to classes
        labels = (labels_raw >= threshold).astype(int)
    else:
        labels = labels_raw  # If regression, use the probabilities as-is
    return labels

def split_dataset(num_entities, train_size, val_size, test_size):
    indices = np.arange(num_entities)
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:train_size + val_size + test_size]

    train_mask = index_to_mask(train_indices, num_entities)
    val_mask = index_to_mask(val_indices, num_entities)
    test_mask = index_to_mask(test_indices, num_entities)

    return train_mask, val_mask, test_mask


def index_to_mask(index, size):
    mask = np.zeros(size)
    mask[index] = 1
    return np.array(mask, dtype=np.bool)
