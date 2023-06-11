import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split

"""
Instead of doing the reshape to 1x9x9 numpy array in memory, it would be faster
(and more hacky) to do it before the training
"""


RAW_PATH = "../data/sudoku/sudoku_pq/sudoku.parquet"
TRAIN_DATA_PATH = "../data/sudoku/sudoku_pq_train_data.csv"
TRAIN_LABEL_PATH = "../data/sudoku/sudoku_pq_train_label.csv"
TEST_DATA_PATH = "../data/sudoku/sudoku_pq_test_data.csv"
TEST_LABEL_PATH = "../data/sudoku/sudoku_pq_test_label.csv"

def train_trans(df):
    return np.array([np.uint(i) for i in df]).reshape((1,9,9))

def test_trans(df):
    return np.array([np.uint(i) for i in df]).reshape((9,9))

def read_data(path):

    df = pd.read_parquet(path)
    df["id"] = range(len(df))
    df["puzzle"] = df["puzzle"].apply(train_trans)
    df["solution"] = df["solution"].apply(test_trans)

    return df

def _train_test_split(data, test_ratio = 0.2):

    X = data["puzzle"]
    y = data["solution"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_ratio, 
                                                        random_state=64)
    return X_train, X_test, y_train, y_test

def fetch_data(path=RAW_PATH):

    print("Reading and converting data...")
    data = read_data(path)
    print("Conversion complete")

    X_train, X_test, y_train, y_test = _train_test_split(data)

    return X_train, X_test, y_train, y_test

def convert_to_nparray(src=RAW_PATH, train_data_dst=TRAIN_DATA_PATH, 
                       train_label_dst=TRAIN_LABEL_PATH,
                       test_data_dst=TEST_DATA_PATH,
                       test_label_dst=TEST_LABEL_PATH):

    print("Saving results as csv file")

    X_train, X_test, y_train, y_test = fetch_data(src)

    df_train_data = pd.DataFrame({"puzzle": X_train})
    df_train_label = pd.DataFrame({"solution": y_train})
    df_test_data = pd.DataFrame({"puzzle": X_test})
    df_test_label = pd.DataFrame({"solution": y_test})

    df_train_data.to_csv(train_data_dst)
    df_train_label.to_csv(train_label_dst)
    df_test_data.to_csv(test_data_dst)
    df_test_label.to_csv(test_label_dst)

if __name__ == "__main__":
    convert_to_nparray()
