import json
import pandas as pd
from pathlib import Path
from typing import Optional


def collect_lightning_logs(logs_dir: str = "lightning_logs") -> pd.DataFrame:
    """
    Collect training results from all version folders in lightning_logs.
    
    Args:
        logs_dir: Path to the lightning_logs directory (default: "lightning_logs")
    
    Returns:
        DataFrame with columns for hyperparameters and final metrics
    """
    logs_path = Path(logs_dir)
    
    if not logs_path.exists():
        raise ValueError(f"Directory {logs_dir} does not exist")
    
    results = []
    
    # Iterate through all version folders
    for version_folder in sorted(logs_path.glob("version_*")):
        if not version_folder.is_dir():
            continue
            
        summary_file = version_folder / "summary.json"
        
        # Skip if summary.json doesn't exist
        if not summary_file.exists():
            continue
        
        try:
            # Read the summary.json file
            with open(summary_file, 'r') as f:
                data = json.load(f)
            
            # Extract relevant information
            row = {
                'version': version_folder.name,
            }
            
            # Add hyperparameters
            if 'hyperparameters' in data:
                hyperparams = data['hyperparameters']
                row['dataset'] = hyperparams.get('dataset')
                row['strategy'] = hyperparams.get('strategy')
                row['strategy_type'] = hyperparams.get('strategy_type')
                row['transition_matrix'] = hyperparams.get('transition_matrix')
                row['seed'] = hyperparams.get('seed')
                row['model'] = hyperparams.get('model')
                row['augment'] = hyperparams.get('data_augment')
           
            # Add final metrics
            if 'final_metrics' in data:
                final_metrics = data['final_metrics']
                row['best_test_acc'] = final_metrics.get('best_test_acc')
                row['best_valid_acc'] = final_metrics.get('best_valid_acc')
                row['final_epoch'] = final_metrics.get('final_epoch')

            if row['final_epoch'] < 99:
                continue  # Skip incomplete runs
            results.append(row)
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse {summary_file}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns to have important ones first
    if not df.empty:
        priority_cols = ['version', 'dataset', 'strategy', 'strategy_type', 
                        'transition_matrix', 'best_test_acc', 'best_valid_acc']
        other_cols = [col for col in df.columns if col not in priority_cols]
        df = df[priority_cols + other_cols]
    
    return df


if __name__ == "__main__":
    # Example usage
    df = collect_lightning_logs()
    print(f"Collected {len(df)} runs")
    print("\nDataFrame preview:")
    print(df.head())
    print("\nDataFrame info:")
    print(df.info())
