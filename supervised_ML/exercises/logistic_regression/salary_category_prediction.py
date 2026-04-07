import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression as SklearnSoftmax

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from supervised_ML.core.softmax_regression import SoftmaxRegression
from utils.data_preprocessing import clean_data

def run_softmax_exercise():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../../data/hr.csv")
    df = clean_data(pd.read_csv(data_path))
    
    # We will predict Salary Category (low=0, medium=1, high=2) based on HR metrics
    df['salary_target'] = df['salary'].map({'low': 0, 'medium': 1, 'high': 2})
    
    # Features: satisfaction, evaluation, promotion
    features = ['satisfaction_level', 'last_evaluation', 'promotion_last_5years', 'average_montly_hours']
    X, y = df[features].values, df['salary_target'].values
    
    # Scaling Features for multi-class optimization stability
    X_max = X.max(axis=0); X_scaled = X / X_max
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.8, random_state=42)
    
    # 1. Custom Softmax (multi-class)
    model = SoftmaxRegression(lr=0.01, iterations=5000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_test == y_pred)
    
    print(f"--- Softmax Multi-Class Performance ---")
    print(f"Custom Softmax Accuracy: {accuracy:.4f}")
    
    # 2. Sklearn Multinomial Baseline
    sk_model = SklearnSoftmax(solver='lbfgs', max_iter=1000).fit(X_train, y_train)
    sk_acc = sk_model.score(X_test, y_test)
    print(f"Sklearn Softmax Accuracy: {sk_acc:.4f}")

    # 3. Visualization: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    categories = ['low', 'medium', 'high']
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted Salary Category')
    plt.ylabel('Actual Salary Category')
    plt.title('Softmax Regression: Confusion Matrix for Salary Prediction')
    
    plots_dir = os.path.join(current_dir, "../../plots/logistic_regression")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "softmax_salary_prediction_matrix.png"))
    print(f"Confusion Matrix Heatmap saved.")

if __name__ == "__main__":
    run_softmax_exercise()
