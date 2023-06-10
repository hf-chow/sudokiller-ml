import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class SudokuDataset(Dataset):
    def __init__(self):
        self.labels = pd.read_parquet(label_file)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data = pd.read_parquet(self.data_dir)[index]
        label = self.labels[index]
        return data, label
