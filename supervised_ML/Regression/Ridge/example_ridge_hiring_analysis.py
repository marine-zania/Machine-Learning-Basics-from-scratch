import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge as SklearnRidge

# Path setup to root for utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from ridge_regression import RidgeRegression
from utils.data_preprocessing import fix_missing_values, clean_data
from utils.encoding import words_to_numbers

def run_ridge_exercise():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../../data/hiring.csv")
    df = clean_data(pd.read_csv(data_path))
    df['experience'] = df['experience'].fillna('zero')
    df = words_to_numbers(df, ['experience'])
    df = fix_missing_values(df, strategy='median')
    
    features = ['experience', 'test_score', 'interview_score']
    X, y = df[features].values, df['salary'].values
    
    # 1. Custom Ridge (alpha=10)
    model = RidgeRegression(alpha=10)
    model.fit(X, y)
    
    cand = [2, 9, 6]
    pred = model.predict([cand]).item()
    print(f"--- Ridge Regression Analysis ---")
    print(f"Candidate {cand} -> Predicted Salary: ${pred:,.2f}")
    
    # 2. Sklearn Ridge Validation
    sk_model = SklearnRidge(alpha=10).fit(X, y)
    if np.allclose(model.predict(X), sk_model.predict(X)):
        print("✅ Custom Ridge matches Sklearn exactly!")

    # 3. Visualization (Individual Feature Trends)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['blue', 'green', 'purple']
    
    for i, feature in enumerate(features):
        axes[i].scatter(df[feature], y, color='red', marker='+', label='Actual')
        temp_model = SklearnRidge(alpha=10).fit(df[[feature]].values, y)
        axes[i].plot(df[feature], temp_model.predict(df[[feature]].values), color=colors[i], label='Ridge Trend')
        axes[i].set_xlabel(feature.capitalize()); axes[i].set_ylabel("Salary ($)")
        axes[i].legend(); axes[i].grid(True)

    plt.suptitle("Ridge Regression (L2): Impact of Experience, Test, and Interview Scores")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(os.path.join(current_dir, "example_ridge_hiring_analysis.png"))
    print(f"Plot saved to: {current_dir}")

if __name__ == "__main__":
    run_ridge_exercise()
