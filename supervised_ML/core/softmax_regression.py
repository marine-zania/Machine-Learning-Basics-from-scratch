import numpy as np

class SoftmaxRegression:
    def __init__(self, lr=0.01, iterations=1000):
        self.lr, self.epochs = lr, iterations
        self.weights, self.bias = None, None

    def fit(self, X, y):
        X, y = np.array(X, dtype=np.float64), np.array(y, dtype=np.int32)
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # OHE for y
        y_ohe = np.eye(n_classes)[y]
        self.weights, self.bias = np.zeros((n_features, n_classes)), np.zeros(n_classes)

        for _ in range(self.epochs):
            scores = np.dot(X, self.weights) + self.bias
            # Softmax calculation
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            
            dw = (1 / n_samples) * np.dot(X.T, (probs - y_ohe))
            db = (1 / n_samples) * np.sum(probs - y_ohe, axis=0)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        scores = np.dot(X, self.weights) + self.bias
        return np.argmax(scores, axis=1)
