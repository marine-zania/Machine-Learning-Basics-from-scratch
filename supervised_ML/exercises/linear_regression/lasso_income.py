import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso as SklearnLasso

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from supervised_ML.core.lasso_regression import LassoRegression
from utils.data_preprocessing import clean_data

def run_lasso_exercise():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../../data/canada_per_capita_income.csv")
    df = clean_data(pd.read_csv(data_path, sep='\t'))
    
    # Feature Scaling (Standardization: X' = (X - mean) / std)
    # This is crucial for Gradient Descent based models like Lasso
    X_raw, y = df['year'].values, df['per capita income (US$)'].values
    mean, std = X_raw.mean(), X_raw.std()
    X_scaled = (X_raw - mean) / std
    
    # 1. Custom Lasso (Optimized for convergence)
    model = LassoRegression(alpha=50, lr=0.1, iterations=5000)
    model.fit(X_scaled, y)
    
    # 2. Sklearn Lasso
    sk_model = SklearnLasso(alpha=50).fit(X_raw.reshape(-1, 1), y)
    
    # 3. Prediction Check (for 2020)
    year_2020_scaled = (2020 - mean) / std
    cust_pred = model.predict(year_2020_scaled)
    sk_pred = sk_model.predict([[2020]]).item()
    
    print("--- Lasso Regression (L1) Canada Income Analysis ---")
    print(f"Custom Predicted (2020): ${cust_pred:,.2f}")
    print(f"Sklearn Predicted (2020): ${sk_pred:,.2f}")
    
    # 4. Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(X_raw, y, color='red', marker='+', label='Actual')
    plt.plot(X_raw, model.predict(X_scaled), color='blue', label='Custom Lasso Fit')
    plt.plot(X_raw, sk_model.predict(X_raw.reshape(-1, 1)), color='green', linestyle='--', label='Sklearn Lasso Fit')
    plt.title("Lasso Regression: Standardized Gradient Descent Results")
    plt.xlabel("Year")
    plt.ylabel("Income ($)")
    plt.legend()
    plt.grid(True)
    
    plots_dir = os.path.join(current_dir, "../../plots/linear_regression")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "lasso_income_analysis.png"))
    print(f"Visualization saved.")

if __name__ == "__main__":
    run_lasso_exercise()
