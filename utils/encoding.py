import pandas as pd

def words_to_numbers(df, columns):
    try:
        from word2number import w2n
        for col in columns:
            df[col] = df[col].apply(lambda x: w2n.word_to_num(x.strip()) if isinstance(x, str) else x)
    except ImportError:
        print("Please install word2number: pip install word2number")
    return df

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
