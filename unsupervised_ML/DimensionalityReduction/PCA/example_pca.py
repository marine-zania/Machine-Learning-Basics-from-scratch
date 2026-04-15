import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as SklearnPCA

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from unsupervised_ML.DimensionalityReduction.PCA.pca import PCA
from utils.data_preprocessing import clean_data

def run_pca_example():
    # Load Iris dataset
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/iris.csv"))
    if not os.path.exists(data_path):
        # Fallback to supervised_ML/data/iris.csv
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../supervised_ML/data/iris.csv"))
        
    df = pd.read_csv(data_path)
    df = clean_data(df)
    
    X = df.drop('species', axis=1).values
    y = df['species'].values
    
    # Custom PCA
    n_components = 2
    model = PCA(n_components=n_components)
    X_projected = model.fit_transform(X)
    
    print(f"--- PCA Dimensionality Reduction (n_components={n_components}) ---")
    print(f"Original shape: {X.shape}")
    print(f"Transformed shape: {X_projected.shape}")
    
    # Sklearn PCA for parity check
    sk_pca = SklearnPCA(n_components=n_components)
    X_sk_projected = sk_pca.fit_transform(X)
    
    # Note: Eigenvectors can have flipped signs between implementations,
    # but the variance captured and the subspace should be the same.
    # We check if the absolute correlation is high.
    correlation = np.abs(np.corrcoef(X_projected[:, 0], X_sk_projected[:, 0])[0, 1])
    print(f"Correlation with Sklearn PC1: {correlation:.4f}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    colors = {'setosa': 'r', 'versicolor': 'g', 'virginica': 'b'}
    
    for species in np.unique(y):
        mask = (y == species)
        plt.scatter(X_projected[mask, 0], X_projected[mask, 1], 
                    c=colors[species], label=species, alpha=0.7)
        
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Iris Dataset')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(os.path.dirname(__file__), "pca_iris_projection.png")
    plt.savefig(plot_path)
    print(f"PCA projection plot saved to {plot_path}")

if __name__ == "__main__":
    run_pca_example()
