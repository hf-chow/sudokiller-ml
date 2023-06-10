from dataloader import SudokuDataset
import torch
from torch.utils.data import Dataloader

TRAIN_DATA_PATH = "../data/sudoku/sudoku_pq_train_data.parquet"
TRAIN_LABEL_PATH = "../data/sudoku/sudoku_pq_train_label.parquet"
TEST_DATA_PATH = "../data/sudoku/sudoku_pq_test_data.parquet"
TEST_LABEL_PATH = "../data/sudoku/sudoku_pq_test_label.parquet"

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

print(f"Using {device} for training")

BATCH_SIZE = 64

train_data = SudokuDataset(data_dir=TRAIN_DATA_PATH, label_dir=TRAIN_LABEL_PATH)
test_data = SudokuDataset(data_dir=TEST_DATA_PATH, label_dir=TEST_LABEL_PATH)

