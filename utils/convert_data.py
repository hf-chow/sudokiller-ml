import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split

RAW_PATH = "../data/sudoku/sudoku_pq/sudoku.parquet"
TRAIN_DATA_PATH = "../data/sudoku_pq/sudoku_pq_train_data.parquet"
TRAIN_LABEL_PATH = "../data/sudoku_pq/sudoku_pq_train_label.parquet"
TEST_DATA_PATH = "../data/sudoku_pq/sudoku_pq_test_data.parquet"
TEST_LABEL_PATH = "../data/sudoku_pq/sudoku_pq_test_label.parquet"

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

def convert_to_nparray(src=RAW_PATH, train_data_dst=TRAIN_DATA_PATH, 
                       train_label_dst=TRAIN_LABEL_PATH,
                       test_data_dst=TEST_DATA_PATH,
                       test_label_dst=TEST_LABEL_PATH):

    print("Saving results as parquet file")

    X_train, X_test, y_train, y_test = fetch_data(src)

    df_train_data = pd.DataFrame({"puzzle": X_train})
    df_train_label = pd.DataFrame({"solution": y_train})
    df_test_data = pd.DataFrame({"puzzle": X_test})
    df_test_label = pd.DataFrame({"solution": y_test})

    df_train_data.to_parquet(train_data_dst)
    df_train_label.to_parquet(train_label_dst)
    df_test_data.to_parquet(test_data_dst)
    df_test_label.to_parquet(test_label_dst)

convert_to_nparray()

