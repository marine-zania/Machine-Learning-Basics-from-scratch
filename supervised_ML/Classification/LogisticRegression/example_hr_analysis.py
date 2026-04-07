import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression as SklearnLR

# No sklearn model_selection imports anymore!
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from logistic_regression import LogisticRegression
from utils.data_preprocessing import clean_data
from utils.encoding import one_hot_encode
from utils.validation import train_test_split_custom

def run_hr_analysis():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../../data/hr.csv")
    
    # 1. Load & Encode (Independent)
    df = clean_data(pd.read_csv(data_path))
    df_encoded = one_hot_encode(df, ['salary'])
    
    # 2. Feature Selection
    features = ['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 
                'salary_low', 'salary_medium', 'salary_high']
    X, y = df_encoded[features].values, df_encoded['left'].values
    
    # 3. Split (Independent)
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)
    
    # 4. Model (Scaled for stability)
    model = LogisticRegression(learning_rate=0.001, iterations=10000)
    X_train_scaled = X_train / X_train.max(axis=0)
    X_test_scaled = X_test / X_train.max(axis=0)
    
    model.fit(X_train_scaled, y_train)
    
    # 5. Result
    y_pred = model.predict(X_test_scaled)
    accuracy = np.mean(y_test == y_pred)
    
    print("--- HR Retention Independent Analysis (No Sklearn Utilities) ---")
    print(f"Dataset Split: {len(X_train)} Train, {len(X_test)} Test")
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")
    
    # 6. Sklearn Baseline Comparison
    sk_model = SklearnLR(max_iter=10000).fit(X_train_scaled, y_train)
    sk_accuracy = sk_model.score(X_test_scaled, y_test)
    print(f"Sklearn Accuracy (Baseline): {sk_accuracy:.4f}")
    
    if np.isclose(accuracy, sk_accuracy, atol=0.05):
        print("✅ Comparison successful: Custom model is near Sklearn baseline.")
    
    # 7. Visualization
    plt.figure(figsize=(10, 6))
    pd.crosstab(df.salary, df.left).plot(kind='bar')
    plt.title("Employee Retention by Salary (Custom Utility Data Check)")
    plt.savefig(os.path.join(current_dir, "example_hr_analysis.png"))
    print(f"\nVisual Analysis Plot generated in: {current_dir}")

if __name__ == "__main__":
    run_hr_analysis()
