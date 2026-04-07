import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier as SklearnRF

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from supervised_ML.core.classification.random_forest import RandomForest
from utils.data_preprocessing import clean_data, one_hot_encode, train_test_split_custom

def run_retention_rf_exercise():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../../data/hr.csv")
    df = clean_data(pd.read_csv(data_path))
    
    # 1. Independent Feature Engineering (Using custom utils)
    print("--- HR Retention Forest: Independent Pre-processing ---")
    df_encoded = one_hot_encode(df, ['salary', 'Department'])
    
    # Selecting all relevant numeric and one-hot features
    X = df_encoded.drop('left', axis=1).values
    y = df_encoded['left'].values
    
    # 2. Independent Split (Using custom utils)
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)
    
    # 3. Custom Forest Training
    print(f"Dataset Split: {len(X_train)} Train, {len(X_test)} Test")
    print("Training Forest (10 trees, Depth 15)...")
    
    model = RandomForest(n_trees=10, max_depth=15)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_test == y_pred)
    print(f"Custom Random Forest Accuracy: {accuracy:.4f}")
    
    # 4. Sklearn for parity verification
    sk_model = SklearnRF(n_estimators=10, max_depth=15, random_state=42)
    sk_model.fit(X_train, y_train)
    print(f"Sklearn Forest Accuracy (Baseline): {sk_model.score(X_test, y_test):.4f}")
    
    # 5. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens', ax=ax1)
    ax1.set_title(f"Custom Independent Forest (Acc: {accuracy:.4f})")
    sns.heatmap(confusion_matrix(y_test, sk_model.predict(X_test)), annot=True, fmt='d', cmap='Purples', ax=ax2)
    ax2.set_title(f"Sklearn Comparison")
    
    plt.suptitle("Independent Random Forest: Employee Retention Study")
    plots_dir = os.path.join(current_dir, "../../plots/random_forest")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "hr_retention_independent.png"))
    print(f"Analysis saved to: {plots_dir}")

if __name__ == "__main__":
    run_retention_rf_exercise()
