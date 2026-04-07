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

def clean_data(df):
    df.columns = [col.strip() for col in df.columns]
    return df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
