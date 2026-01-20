import pytorch_lightning as pl
import os
import csv
import json
from collections import defaultdict


class MetricsLogger(pl.Callback):
    """Callback to log metrics to a CSV file"""
    
    def __init__(self, log_dir, filename="metrics.csv"):
        super().__init__()
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, filename)
        self.metrics_buffer = defaultdict(list)
        self.current_epoch_metrics = {}
        
    def on_train_start(self, trainer, pl_module):
        """Initialize CSV file with headers"""
        os.makedirs(self.log_dir, exist_ok=True)
        
    def on_train_epoch_end(self, trainer, pl_module):
        """Log training metrics at end of epoch"""
        if trainer.sanity_checking:
            return
        
        # Collect all metrics from callback_metrics
        epoch_data = {
            'epoch': trainer.current_epoch,
        }
        
        # Add all logged metrics
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, (int, float)):
                epoch_data[key] = float(value)
            elif hasattr(value, 'item'):
                epoch_data[key] = float(value.item())
        
        # Write to CSV
        self._write_to_csv(epoch_data)
    
    def _write_to_csv(self, data):
        """Write metrics to CSV file"""
        file_exists = os.path.isfile(self.log_file)
        
        # Ensure epoch and step are first columns, then sort the rest
        fieldnames = ['epoch']
        other_fields = sorted([k for k in data.keys() if k not in ['epoch']])
        fieldnames.extend(other_fields)
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(data)


class SummaryLogger(pl.Callback):
    """Callback to save final summary as JSON"""
    
    def __init__(self, log_dir, hparams, filename="summary.json"):
        super().__init__()
        self.log_dir = log_dir
        self.summary_file = os.path.join(log_dir, filename)
        self.hparams = hparams
        self.best_test_acc = 0.0
        self.best_valid_acc = 0.0
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """Track best accuracies and save summary"""
        if trainer.sanity_checking:
            return
        
        metrics = trainer.callback_metrics
        
        if 'Best_Test_Acc' in metrics:
            self.best_test_acc = float(metrics['Best_Test_Acc'].item())
        
        if 'Valid_Accuracy' in metrics:
            valid_acc = float(metrics['Valid_Accuracy'].item())
            if valid_acc > self.best_valid_acc:
                self.best_valid_acc = valid_acc
        
        # Save summary after each validation to handle early termination
        self._save_summary(trainer, metrics)
    
    def on_train_end(self, trainer, pl_module):
        """Save final summary as JSON"""
        metrics = trainer.callback_metrics
        self._save_summary(trainer, metrics, is_final=True)
        
        print(f"\nSummary saved to: {self.summary_file}")
        print(f"Best Test Acc: {self.best_test_acc:.6f}")
        # print(f"Best Valid Acc: {self.best_valid_acc:.6f}")
    
    def on_exception(self, trainer, pl_module, exception):
        """Save summary even if training crashes"""
        try:
            metrics = trainer.callback_metrics
            self._save_summary(trainer, metrics, is_final=False)
            print(f"\nSummary saved on exception to: {self.summary_file}")
        except Exception as e:
            print(f"Failed to save summary on exception: {e}")
    
    def _save_summary(self, trainer, metrics, is_final=False):
        """Save summary to JSON file"""
        summary = {
            "hyperparameters": self.hparams,
            "final_metrics": {
                "final_epoch": trainer.current_epoch,
                "final_step": trainer.global_step,
                "best_test_acc": self.best_test_acc * 100,
                "best_valid_acc": self.best_valid_acc * 100,
                "training_completed": is_final,
            },
            "latest_metrics": {}
        }
        
        # Add all final metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                summary["latest_metrics"][key] = float(value)
            elif hasattr(value, 'item'):
                summary["latest_metrics"][key] = float(value.item())
        
        # Save to JSON
        os.makedirs(self.log_dir, exist_ok=True)
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

