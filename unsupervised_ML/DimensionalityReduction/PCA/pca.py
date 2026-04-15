import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA) implemented from scratch.
    
    Parameters:
    -----------
    n_components : int
        Number of principal components to keep.
    """
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None

    def fit(self, X):
        """
        Fit the model with X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        """
        X = np.array(X)
        # Mean centring
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        
        # Covariance matrix (row = sample, col = feature)
        # np.cov expects features in rows, so we transpose
        cov = np.cov(X.T)
        
        # Eigenvectors, Eigenvalues
        # eigenvalues: (n_features,), eigenvectors: (n_features, n_features)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Eigenvectors are returned as columns, transpose for easier sorting/access
        eigenvectors = eigenvectors.T
        
        # Sort eigenvectors by eigenvalues descending
        idxs = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idxs]
        self.components = eigenvectors[idxs]
        
        # Store only the requested number of components
        self.components = self.components[:self.n_components]

    def transform(self, X):
        """
        Apply dimensionality reduction to X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            New data.
            
        Returns:
        --------
        X_transformed : ndarray of shape (n_samples, n_components)
        """
        X = np.array(X)
        X = X - self.mean
        return np.dot(X, self.components.T)

    def fit_transform(self, X):
        """
        Fit the model with X and apply the dimensionality reduction on X.
        """
        self.fit(X)
        return self.transform(X)
