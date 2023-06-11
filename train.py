from model import CNN
from dataloader import SudokuDataset
import torch
from torch.utils.data import DataLoader

TRAIN_DATA_PATH = "../data/sudoku/sudoku_pq_train_data.csv"
TRAIN_LABEL_PATH = "../data/sudoku/sudoku_pq_train_label.csv"
TEST_DATA_PATH = "../data/sudoku/sudoku_pq_test_data.csv"
TEST_LABEL_PATH = "../data/sudoku/sudoku_pq_test_label.csv"

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

print(f"Using {device} for training")

BATCH_SIZE = 64

train_data = SudokuDataset(data_dir=TRAIN_DATA_PATH, label_dir=TRAIN_LABEL_PATH)
test_data = SudokuDataset(data_dir=TEST_DATA_PATH, label_dir=TEST_LABEL_PATH)

model = CNN().to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

def train(dataloader, model, loss_fn, optim):
    size = len(dataloader.dataset)
    model.train()
    for batch, data in enumerate(dataloader):
        X, y = data[0].to(device), data[1].to(device)
        pred = model(X)
        loss = loss_fn(pred, X)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if batch%100 == 0:
            loss, current = loss.item(), (batch+1)*len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

train(train_dataloader, model, loss_fn, optim)
