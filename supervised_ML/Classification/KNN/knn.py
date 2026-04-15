import numpy as np
from collections import Counter

class KNN:
    """
    K-Nearest Neighbors Classifier implemented from scratch.
    
    Parameters:
    -----------
    k : int, default=3
        Number of nearest neighbors to consider.
    """
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target values.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        """
        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y)

    def predict(self, X):
        """
        Predict the class labels for the provided data.
        
        Parameters:
        -----------
        X : array-like of shape (n_queries, n_features)
            Test samples.
            
        Returns:
        --------
        y_pred : ndarray of shape (n_queries,)
            Class labels for each data sample.
        """
        X = np.array(X, dtype=np.float64)
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

    def _predict_single(self, x):
        # Compute distances between x and all examples in the training set
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _euclidean_distance(self, x1, x2):
        """
        Helper method to compute Euclidean distance between two points.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
