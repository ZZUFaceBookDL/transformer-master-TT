import json
from typing import Optional
import torch
import numpy as np
from torch.utils.data import Dataset


class OzeDataset(Dataset):
    def __init__(self, filepath):
        super(OzeDataset, self).__init__()
        xy = np.loadtxt(filepath, delimiter='\t', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = xy[:, 1:]
        self.y_data = xy[:, 0]

        self.min_label = min(self.y_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index] - self.min_label

    def __len__(self):
        return self.len