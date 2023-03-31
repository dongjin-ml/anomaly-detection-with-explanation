import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    
    def __init__(self, x, y):
        
        self.x, self.y = x, y

    def __len__(self):
        
        return len(self.x)

    def __getitem__(self, idx):
        
        x = torch.tensor(self.x[idx]).type(torch.float32)
        y = torch.tensor(self.y[idx]).type(torch.int)
        
        return x, y