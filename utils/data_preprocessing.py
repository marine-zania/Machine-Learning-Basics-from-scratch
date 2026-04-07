import pandas as pd
import numpy as np

def fix_missing_values(df, strategy='mean'):
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            if strategy == 'mean': fill_val = df[col].mean()
            elif strategy == 'median': fill_val = df[col].median()
            else: fill_val = 0
            df[col] = df[col].fillna(fill_val)
    return df

def words_to_numbers(df, columns):
    try:
        from word2number import w2n
        for col in columns:
            df[col] = df[col].apply(lambda x: w2n.word_to_num(x.strip()) if isinstance(x, str) else x)
    except ImportError:
        print("Please install word2number: pip install word2number")
    return df

def clean_data(df):
    df.columns = [col.strip() for col in df.columns]
    return df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

def one_hot_encode(df, columns):
    """Manual One-Hot Encoding implementation."""
    df_copy = df.copy()
    for col in columns:
        # Get unique categories
        categories = df_copy[col].unique()
        for cat in categories:
            # Create a 1/0 column for each category
            df_copy[f"{col}_{cat}"] = (df_copy[col] == cat).astype(int)
        # Drop the original categorical column
        df_copy.drop(col, axis=1, inplace=True)
    return df_copy

def train_test_split_custom(X, y, test_size=0.2, random_state=None):
    """Manual Train-Test Split with randomization."""
    if random_state: np.random.seed(random_state)
    
    # Shuffle indices
    idxs = np.arange(len(X))
    np.random.shuffle(idxs)
    
    # Split point
    split_idx = int(len(X) * (1 - test_size))
    
    train_idxs = idxs[:split_idx]
    test_idxs = idxs[split_idx:]
    
    return X[train_idxs], X[test_idxs], y[train_idxs], y[test_idxs]
