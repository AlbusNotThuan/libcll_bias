import pytorch_lightning as pl
import torch


class TestEpochCallback(pl.Callback):
    """Callback to evaluate on test set at each validation epoch"""
    
    def __init__(self, test_loader):
        super().__init__()
        self.test_loader = test_loader
        self.best_test_acc = 0.0
        self.best_valid_acc = 0.0
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Run test evaluation after each validation epoch"""
        if trainer.sanity_checking:
            return
        
        # Use the strategy's actual test_step method to ensure consistency
        was_training = pl_module.training
        pl_module.eval()
        
        # Clear test_acc list before evaluation
        pl_module.test_acc.clear()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                x, y = batch
                x, y = x.to(pl_module.device), y.to(pl_module.device)
                # Call the strategy's test_step method directly
                pl_module.test_step((x, y), batch_idx)
        
        # Aggregate using the same method as on_test_epoch_end
        avg_test_acc = torch.stack(pl_module.test_acc).mean()
        
        # Clear the test_acc list after use
        pl_module.test_acc.clear()
        
        # Track best test accuracy
        if avg_test_acc > self.best_test_acc:
            self.best_test_acc = avg_test_acc
        
        # Track best validation accuracy
        if 'Valid_Accuracy' in trainer.callback_metrics:
            valid_acc = trainer.callback_metrics['Valid_Accuracy']
            if valid_acc > self.best_valid_acc:
                self.best_valid_acc = valid_acc
        
        # Log with same names - Test_Acc for per-epoch, matching test_step behavior
        pl_module.log("Test_Acc", avg_test_acc, prog_bar=True, sync_dist=True)
        pl_module.log("Best_Test_Acc", self.best_test_acc, prog_bar=True, sync_dist=True)
        
        # Log best metrics to hyperparameters
        if trainer.logger:
            trainer.logger.log_metrics({
                "hp/best_test_acc": self.best_test_acc * 100,
                "hp/best_valid_acc": self.best_valid_acc * 100,
                "hp/current_epoch": trainer.current_epoch,
            }, step=trainer.global_step)
        
        # Restore original training mode
        if was_training:
            pl_module.train()

