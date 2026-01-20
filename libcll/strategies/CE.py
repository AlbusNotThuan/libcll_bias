import torch
import torch.nn.functional as F
from .Strategy import Strategy


class CE(Strategy):
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = self.compute_ce(out, y)
        self.log("Train_Loss", loss)
        return loss
