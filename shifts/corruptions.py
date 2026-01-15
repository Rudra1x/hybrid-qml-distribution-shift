import numpy as np
import torch
from torch.utils.data import Dataset

class CIFAR10C(Dataset):
    def __init__(self, corruption, severity, root="./data/cifar10c"):
        self.data = np.load(f"{root}/{corruption}.npy")
        self.labels = np.load(f"{root}/labels.npy")

        start = (severity - 1) * 10000
        end = severity * 10000

        self.data = self.data[start:end]
        self.labels = self.labels[start:end]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx].astype("float32") / 255.0
        x = torch.tensor(x).permute(2, 0, 1)
        y = torch.tensor(self.labels[idx])
        return x, y