import pandas as pd
import datasets
from sklearn.model_selection import train_test_split

def get_raw_data(csv_name, label_col_name="gen_label", text_col_name="Messages"):
    df = pd.read_csv(f"raw_data/{csv_name}")
    df["text"] = df[text_col_name]
    df["label"] = df[label_col_name]
    df = df.dropna()
    df = df.drop_duplicates()
    print(f"""raw data breakdown by label:\n
          {df[label_col_name].value_counts()}
        """)
    return df.drop(columns=[label_col_name, text_col_name]) 

def train_val_test_split(df, label_col_name="label", train_size=0.8, has_val=True):
    """Return a tuple (DataFrame, DatasetDict) with a custom train/val/split"""

    if isinstance(train_size, int):
        train_size = train_size / len(df)

    df = df.sample(frac=1, random_state=0)
    df_train, df_test = train_test_split(
        df, test_size=1 - train_size, stratify=df[label_col_name]
    )

    if has_val:
        df_test, df_val = train_test_split(
            df_test, test_size=0.5, stratify=df_test[label_col_name]
        )
        return (
            (df_train, df_val, df_test),
            datasets.DatasetDict(
                {
                    "train": datasets.Dataset.from_pandas(df_train),
                    "val": datasets.Dataset.from_pandas(df_val),
                    "test": datasets.Dataset.from_pandas(df_test),
                }
            ),
        )

    else:
        return (
            (df_train, df_test),
            datasets.DatasetDict(
                {
                    "train": datasets.Dataset.from_pandas(df_train),
                    "test": datasets.Dataset.from_pandas(df_test),
                }
            ),
        )