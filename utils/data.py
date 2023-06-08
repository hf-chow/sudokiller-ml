import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split

RAW_PATH = "../data/sudoku_pq/sudoku.parquet"

def data_trans(df):
    return np.array([np.uint(i) for i in df])

def read_data(path):

    df = pd.read_parquet(path)
    df = df.apply(data_trans, axis=1)

    return df

def _train_test_split(data, test_ratio = 0.2):

    X = data["puzzle"]
    y = data["solution"]

    X_train, X_test, y_train, y_test = _train_test_split(X, y, 
                                                        test_size=test_ratio, 
                                                        random_state=64)
    return X_train, X_test, y_train, y_test

def fetch_data(path=RAW_PATH):

    print("Reading and converting data...")
    data = read_data(path)
    print("Reading complete")

    X_train, X_test, y_train, y_test = _train_test_split(data)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = fetch_data()

