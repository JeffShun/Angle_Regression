import torch.nn as nn

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, predictions, labels):
        return {"mse": self.mse(predictions, labels)}