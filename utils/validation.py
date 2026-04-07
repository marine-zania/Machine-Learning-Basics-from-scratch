import numpy as np

def train_test_split_custom(X, y, test_size=0.2, random_state=None):
    """Manual Train-Test Split with randomization."""
    if random_state: 
        np.random.seed(random_state)
    
    # Shuffle indices
    idxs = np.arange(len(X))
    np.random.shuffle(idxs)
    
    # Split point
    split_idx = int(len(X) * (1 - test_size))
    
    train_idxs = idxs[:split_idx]
    test_idxs = idxs[split_idx:]
    
    return X[train_idxs], X[test_idxs], y[train_idxs], y[test_idxs]


def k_fold_split(X, y, k=5, shuffle=True, random_state=None):
    """
    Generates K-fold training and testing splits.
    
    Args:
        X: Dataset features (numpy array).
        y: Target variables (numpy array).
        k: Number of folds.
        shuffle: Whether to shuffle the data before splitting.
        random_state: Seed for the random number generator.
        
    Yields:
        (X_train, X_test, y_train, y_test) for each fold.
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
        
    # Calculate fold sizes (distribute remainder to the first few folds)
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1
    
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_indices = indices[start:stop]
        
        # Everything not in the test_indices is the training set
        train_indices = np.concatenate((indices[:start], indices[stop:]))
        
        yield X[train_indices], X[test_indices], y[train_indices], y[test_indices]
        current = stop
