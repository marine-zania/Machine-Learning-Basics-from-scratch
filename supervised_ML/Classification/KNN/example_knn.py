import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from supervised_ML.Classification.KNN.knn import KNN
from utils.data_preprocessing import clean_data
from utils.validation import train_test_split_custom

def run_knn_example():
    # Load Iris dataset
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/iris.csv"))
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    df = clean_data(df)
    
    # Let's use all features for a more realistic accuracy comparison
    # But for plotting we might want to restrict to 2 features later
    X = df.drop('species', axis=1).values
    # Map species to integers
    unique_species = df['species'].unique()
    species_to_idx = {species: idx for idx, species in enumerate(unique_species)}
    y = df['species'].map(species_to_idx).values
    
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)
    
    # Custom KNN
    k = 5
    model = KNN(k=k)
    model.fit(X_train, y_train)
    y_pred_custom = model.predict(X_test)
    custom_accuracy = np.mean(y_pred_custom == y_test)
    
    # Sklearn KNN
    sk_model = SklearnKNN(n_neighbors=k)
    sk_model.fit(X_train, y_train)
    sk_accuracy = sk_model.score(X_test, y_test)
    
    print(f"--- KNN Classification (k={k}) ---")
    print(f"Custom KNN Accuracy:  {custom_accuracy:.4f}")
    print(f"Sklearn KNN Accuracy: {sk_accuracy:.4f}")
    
    # Binary classification plot (Setosa vs others) for visualization (using 2 features)
    df_visual = df.copy()
    df_visual['is_setosa'] = df_visual['species'].apply(lambda x: 1 if x == 'setosa' else 0)
    X_v = df_visual[['sepal_length', 'sepal_width']].values
    y_v = df_visual['is_setosa'].values
    
    X_v_train, X_v_test, y_v_train, y_v_test = train_test_split_custom(X_v, y_v, test_size=0.2, random_state=42)
    
    model_v = KNN(k=k)
    model_v.fit(X_v_train, y_v_train)
    
    plt.figure(figsize=(10, 6))
    
    # Create a mesh grid to plot decision boundary
    x_min, x_max = X_v[:, 0].min() - 0.5, X_v[:, 0].max() + 0.5
    y_min, y_max = X_v[:, 1].min() - 0.5, X_v[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    Z = model_v.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X_v[y_v == 1, 0], X_v[y_v == 1, 1], c='green', marker='+', label='Setosa')
    plt.scatter(X_v[y_v == 0, 0], X_v[y_v == 0, 1], c='blue', marker='.', label='Others')
    
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title(f'KNN Decision Boundary (k={k}): Setosa vs Others')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(os.path.dirname(__file__), "knn_iris_boundary.png")
    plt.savefig(plot_path)
    print(f"Decision boundary plot saved to {plot_path}")

if __name__ == "__main__":
    run_knn_example()
