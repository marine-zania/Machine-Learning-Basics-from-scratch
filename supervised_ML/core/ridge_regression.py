import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha, self.weights, self.bias = alpha, None, None

    def fit(self, X, y):
        X, y = np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)
        if X.ndim == 1: X = X.reshape(-1, 1)
        n_features = X.shape[1]
        
        # Add bias column
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Identity matrix for regularization (excluding bias term)
        I = np.identity(n_features + 1)
        I[0, 0] = 0 
        
        # theta = (X^T * X + alpha * I)^(-1) * X^T * y
        theta = np.linalg.inv(X_b.T.dot(X_b) + self.alpha * I).dot(X_b.T).dot(y)
        self.bias, self.weights = theta[0], theta[1:]

    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1: X = X.reshape(-1, 1)
        return X.dot(self.weights) + self.bias
