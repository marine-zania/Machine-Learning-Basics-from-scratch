import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier as SklearnRF

# Path setup to root for utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from random_forest import RandomForest
from utils.data_preprocessing import clean_data
from utils.encoding import one_hot_encode
from utils.validation import train_test_split_custom

def run_retention_rf_exercise():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../../data/hr.csv")
    df = clean_data(pd.read_csv(data_path))
    
    print("--- HR Retention Forest: Independent Pre-processing ---")
    df_encoded = one_hot_encode(df, ['salary', 'Department'])
    X, y = df_encoded.drop('left', axis=1).values, df_encoded['left'].values
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)
    
    print(f"Dataset Split: {len(X_train)} Train, {len(X_test)} Test")
    model = RandomForest(n_trees=10, max_depth=15)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_test == y_pred)
    print(f"Custom Random Forest Accuracy: {accuracy:.4f}")
    
    sk_model = SklearnRF(n_estimators=10, max_depth=15, random_state=42).fit(X_train, y_train)
    print(f"Sklearn Forest Accuracy (Baseline): {sk_model.score(X_test, y_test):.4f}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens', ax=ax1)
    ax1.set_title(f"Custom Forest (Acc: {accuracy:.4f})")
    sns.heatmap(confusion_matrix(y_test, sk_model.predict(X_test)), annot=True, fmt='d', cmap='Purples', ax=ax2)
    ax2.set_title(f"Sklearn Comparison")
    
    plt.suptitle("Independent Random Forest: Employee Retention Study")
    plt.savefig(os.path.join(current_dir, "example_hr_retention.png"))
    print(f"Analysis saved to: {current_dir}")

if __name__ == "__main__":
    run_retention_rf_exercise()
