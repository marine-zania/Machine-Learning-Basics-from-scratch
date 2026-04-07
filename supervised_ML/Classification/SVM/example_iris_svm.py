import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC as SklearnSVM

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from svm import SVM
from utils.data_preprocessing import clean_data
from utils.validation import train_test_split_custom

def run_iris_svm():
    # Data is in root data/ iris.csv
    data_path = os.path.dirname(__file__) + "/../../data/iris.csv"
    df = clean_data(pd.read_csv(data_path))
    
    df['is_setosa'] = df['species'].apply(lambda x: 1 if x == 'setosa' else -1)
    X, y = df[['sepal_length', 'sepal_width']].values, df['is_setosa'].values
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)
    
    model = SVM(learning_rate=0.001, iterations=1000)
    model.fit(X_train, y_train)
    
    print(f"--- Iris SVM Analysis ---")
    print(f"Custom SVM Accuracy: {np.mean(y_test == model.predict(X_test)):.4f}")
    
    sk_model = SklearnSVM(kernel='linear', C=1.0).fit(X_train, y_train)
    print(f"Sklearn SVM Accuracy: {sk_model.score(X_test, y_test):.4f}")
    
    # Plot saved in CURRENT folder
    plt.figure(figsize=(10, 6))
    plt.scatter(df[df.is_setosa==1]['sepal_length'], df[df.is_setosa==1]['sepal_width'], color='green', marker='+', label='Setosa')
    plt.scatter(df[df.is_setosa==-1]['sepal_length'], df[df.is_setosa==-1]['sepal_width'], color='blue', marker='.', label='Others')
    
    x1 = np.linspace(4, 8, 100)
    plt.plot(x1, (model.bias - model.weights[0]*x1)/model.weights[1], color='red', label='Boundary')
    
    plt.title('SVM Decision Boundary: Setosa vs Others'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.dirname(__file__) + "/example_iris_boundary.png")

if __name__ == "__main__":
    run_iris_svm()
