import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        if X.ndim == 1: X = X.reshape(-1, 1)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.bias, self.weights = theta[0], theta[1:]

    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1: X = X.reshape(-1, 1)
        return X.dot(self.weights) + self.bias

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../data/canada_per_capita_income.csv")
    
    # Load dataset (TSV format)
    df = pd.read_csv(data_path, sep='\t')
    df.columns = [col.strip() for col in df.columns]
    
    feature_col = 'year'
    target_col = 'per capita income (US$)'
    
    print(f"Dataset Info:\n{df.head()}\n" + "-" * 30)
    X, y = df[[feature_col]].values, df[target_col].values
    
    # Custom Model
    model = LinearRegression()
    model.fit(X, y)
    
    predict_year = 2020
    custom_pred = model.predict([[predict_year]])[0]
    print(f"--- Custom Model ---\nIntercept: {model.bias:.4f}\nCoefficient: {model.weights[0]:.4f}")
    print(f"Predicted income for {predict_year}: ${custom_pred:.2f}")
    
    # Sklearn Verification
    sk_model = SklearnLinearRegression().fit(X, y)
    sk_pred = sk_model.predict([[predict_year]])[0]
    print(f"\n--- Sklearn Model ---\nIntercept: {sk_model.intercept_:.4f}\nCoefficient: {sk_model.coef_[0]:.4f}")
    print(f"Predicted income for {predict_year}: ${sk_pred:.2f}\n" + "-" * 30)
    
    if np.allclose([model.bias, model.weights[0]], [sk_model.intercept_, sk_model.coef_[0]]):
        print("Success! Custom implementation matches sklearn.")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(df[feature_col], df[target_col], color='red', marker='+', label='Actual Data')
    plt.plot(df[feature_col], model.predict(X), color='blue', label='Regression Line')
    plt.xlabel("Year"), plt.ylabel("Income (US$)"), plt.title("Canada Income Prediction"), plt.legend(), plt.grid(True)
    
    plots_dir = os.path.join(current_dir, "../plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "canada_income_prediction.png"))
    print(f"\nPlot saved to: {os.path.abspath(os.path.join(plots_dir, 'canada_income_prediction.png'))}")

if __name__ == "__main__":
    main()
