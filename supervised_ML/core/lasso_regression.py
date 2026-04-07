import numpy as np

class LassoRegression:
    def __init__(self, alpha=1.0, lr=0.01, iterations=1000):
        self.alpha, self.lr, self.epochs = alpha, lr, iterations
        self.m, self.b = 0, 0

    def fit(self, x, y):
        n = len(x)
        # Using Coordinate Descent style for Lasso convergence
        for _ in range(self.epochs):
            y_pred = self.m * x + self.b
            
            # Gradients with L1 penalty sign(m)
            dm = (1/n) * (sum(x * (y_pred - y)) + self.alpha * np.sign(self.m))
            db = (1/n) * sum(y_pred - y)
            
            self.m -= self.lr * dm
            self.b -= self.lr * db

    def predict(self, x):
        return self.m * x + self.b
