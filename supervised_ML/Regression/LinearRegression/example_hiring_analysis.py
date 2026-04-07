import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SklearnLR

# Path setup to root for utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from linear_regression import LinearRegression
from utils.data_preprocessing import fix_missing_values, clean_data
from utils.encoding import words_to_numbers

def run_hiring_exercise():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../../data/hiring.csv")
    df = clean_data(pd.read_csv(data_path))
    
    # Preprocess
    df['experience'] = df['experience'].fillna('zero')
    df = words_to_numbers(df, ['experience'])
    df = fix_missing_values(df, strategy='median')
    
    features = ['experience', 'test_score', 'interview_score']
    X, y = df[features].values, df['salary'].values
    
    # 1. Custom Model
    model = LinearRegression()
    model.fit(X, y)
    
    candidates = [[2, 9, 6], [12, 10, 10]]
    print("--- Hiring Salary Analysis (Multivariate) ---")
    for cand in candidates:
        pred = model.predict([cand]).item()
        print(f"Candidate {cand} -> Estimated Salary: ${pred:,.2f}")
    
    # 2. Validation with Sklearn
    sk_model = SklearnLR().fit(X, y)
    if np.allclose(model.predict(X), sk_model.predict(X)):
        print("\n✅ Success! Custom Model matches Sklearn results.")
    
    # 3. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['blue', 'green', 'orange']
    for i, feature in enumerate(features):
        axes[i].scatter(df[feature], y, color='red', marker='+', label='Actual')
        temp_model = SklearnLR().fit(df[[feature]].values, y)
        axes[i].plot(df[feature], temp_model.predict(df[[feature]].values), color=colors[i], label='Trend')
        axes[i].set_xlabel(feature.capitalize()); axes[i].set_ylabel("Salary ($)")
        axes[i].set_title(f"{feature.capitalize()} vs Salary"); axes[i].legend(); axes[i].grid(True)

    plt.suptitle("Impact of Experience, Test Scores, and Interview Scores on Salary")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(os.path.join(current_dir, "example_hiring_analysis.png"))
    print(f"\nVisualization saved to: {current_dir}")

if __name__ == "__main__":
    run_hiring_exercise()
