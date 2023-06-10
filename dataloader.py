import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class SudokuDataset(Dataset):
    def __init__(self, data_dir, label_dir):
        self.labels = pd.read_parquet(label_dir)
        self.data = pd.read_parquet(data_dir)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        return data, label
