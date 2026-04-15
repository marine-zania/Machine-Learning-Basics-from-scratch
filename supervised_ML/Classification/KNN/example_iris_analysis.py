import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN

# Path setup to root for utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from supervised_ML.Classification.KNN.knn import KNN
from utils.data_preprocessing import clean_data
from utils.validation import train_test_split_custom

def run_iris_analysis():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../../data/iris.csv")
    
    # 1. Load Data
    df = clean_data(pd.read_csv(data_path))
    
    # 2. Feature Selection
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X, y = df[features].values, df['species'].values
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)
    
    print("--- Iris Flower Classification Analysis (KNN) ---")
    
    # 4. Custom KNN Model
    model = KNN(k=3)
    model.fit(X_train, y_train)
    
    # 5. Result
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    
    print(f"Dataset Split: {len(X_train)} Train, {len(X_test)} Test")
    print(f"Custom KNN Accuracy: {(accuracy * 100):.2f}%")
    
    # 6. Sklearn Baseline Comparison
    sk_model = SklearnKNN(n_neighbors=3).fit(X_train, y_train)
    sk_accuracy = sk_model.score(X_test, y_test)
    print(f"Sklearn KNN Accuracy (Baseline): {(sk_accuracy * 100):.2f}%")
    
    if np.isclose(accuracy, sk_accuracy, atol=0.05):
        print("✅ Comparison successful: Custom model matches Sklearn baseline.")
    
    # 7. Visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species', palette='viridis')
    plt.title("Iris Species Distribution (Petal Length vs Width)")
    plt.grid(True)
    plt.savefig(os.path.join(current_dir, "example_iris_analysis.png"))
    print(f"\nVisual Analysis Plot generated in: {current_dir}")

    # 8. Sample Predictions
    print(f"\n--- Sample Predictions ---")
    sample_indices = [0, 10, 20]
    for idx in sample_indices:
        test_sample = X_test[idx]
        true_class = y_test[idx]
        pred_class = model.predict([test_sample])[0]
        print(f"Sample {idx} -> Predicted: {pred_class}, True: {true_class}")

if __name__ == "__main__":
    run_iris_analysis()
