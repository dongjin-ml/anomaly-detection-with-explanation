import torch.nn as nn

class AutoEncoder(nn.Module):
    
    def __init__(self, input_dim):
        
        super(AutoEncoder, self).__init__()
        self.Encoder_1 = nn.Sequential(
            nn.Linear(input_dim,64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU()
        )
        self.Encoder_2 = nn.Sequential(
            nn.Linear(64,32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU()
        )
        
        self.Decoder = nn.Sequential(
            nn.Linear(32,64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64,input_dim),
        )
        
    def forward(self, x, hidden=False):
        
        if not hidden:
            x = self.Encoder_1(x)
            x = self.Encoder_2(x)
            x = self.Decoder(x)
            
            return x
        
        else:
            e1 = self.Encoder_1(x)
            e2 = self.Encoder_2(e1)
        
            return e1, e2