from torch import nn, flatten

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3),
                nn.Conv2d(64, 128, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3),
                nn.Conv2d(128, 256, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3),
                nn.Conv2d(256, 512, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3)
                )

    def forward(self, x):
        x = self.net(x)
        return x
