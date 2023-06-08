import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split

RAW_PATH = "../data/sudoku_pq/sudoku.parquet"
TRAIN_PATH = "../data/sudoku_pq_train/sudoku_pq_train.parquet"
TEST_PATH = "../data/sudoku_pq_test/sudoku_pq_test.parquet"


def data_trans(df):
    return np.array([np.uint(i) for i in df])

def read_data(path):

    df = pd.read_parquet(path)
    df = df.applymap(data_trans)

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

def convert_to_nparray(src=RAW_PATH, train_dst=TRAIN_PATH, test_dst=TEST_PATH):

    print("Saving results as parquet file")

    X_train, X_test, y_train, y_test = fetch_data(src)

    df_train = pd.DataFrame({"puzzle": X_train, "solution": y_train})
    df_test = pd.DataFrame({"puzzle": X_test, "solution": y_test})

    df_train.to_parquet(train_dst)
    df_test.to_parquet(test_dst)

convert_to_nparray()
