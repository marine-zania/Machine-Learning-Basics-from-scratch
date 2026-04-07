import pandas as pd
import numpy as np

def fix_missing_values(df, strategy='mean'):
    """
    Fills NaN values in numeric columns.
    strategy: 'mean', 'median', or 'zero'
    """
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            if strategy == 'mean':
                fill_val = df[col].mean()
            elif strategy == 'median':
                fill_val = df[col].median()
            else:
                fill_val = 0
            df[col] = df[col].fillna(fill_val)
    return df

def words_to_numbers(df, columns):
    """
    Converts standard English number words to integers.
    """
    try:
        from word2number import w2n
        for col in columns:
            df[col] = df[col].apply(lambda x: w2n.word_to_num(x.strip()) if isinstance(x, str) else x)
    except ImportError:
        print("Please install word2number: pip install word2number")
    return df

def clean_data(df):
    """Basic structural cleaning: stripping whitespace from headers and values."""
    df.columns = [col.strip() for col in df.columns]
    # Strip whitespace from every string cell in the dataframe
    return df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

if __name__ == "__main__":
    # Example logic
    print("Pre-processing utility library ready.")
