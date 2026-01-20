# Chou, Yu-Ting, et al. "Unbiased risk estimators can mislead: A case study of learning with complementary labels." International Conference on Machine Learning. PMLR, 2020.
import torch
import torch.nn.functional as F
from libcll.strategies.Strategy import Strategy


class SCL(Strategy):
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        if self.type == "NL":
            p = (1 - F.softmax(out, dim=1) + 1e-6).log() * -1
            loss = -F.nll_loss(p, y.long())
        elif self.type == "EXP":
            p = torch.exp(F.softmax(out, dim=1))
            loss = -F.nll_loss(p, y.long())
        elif self.type == "FWD":
            p = torch.mm(F.softmax(out, dim=1), self.Q.to(out.device)) + 1e-6
            loss = F.nll_loss(p.log(), y.long())
        else:
            raise NotImplementedError(
                'The type of SCL must be chosen from "NL", "EXP" or "FWD".'
            )
        
        # Compute and log training accuracy
        train_acc = self.compute_acc(out, y)
        
        self.log("Train_Loss", loss)
        self.log("Train_Acc", train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss
