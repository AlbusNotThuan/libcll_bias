from time import sleep
import torch
import numpy as np
import os
import sys
from argparse import ArgumentParser
from omegacli import parse_config, OmegaConf
from libcll.models import build_model
from libcll.strategies import build_strategy
from libcll.datasets import prepare_cl_data_module
from libcll.callbacks import TestEpochCallback
from libcll.metrics_logger import MetricsLogger, SummaryLogger
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI
import random


def main(args):
    sleep(np.random.rand() * 5)  # To avoid possible file write conflicts in multi-process scenarios
    print("Preparing Dataset......")
    pl.seed_everything(args.training.seed, workers=True)
    
    # Set deterministic behavior for reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Required for deterministic CuBLAS
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)
    
    cl_data_module = prepare_cl_data_module(
        args.dataset._name,
        batch_size=args.training.batch_size,
        valid_split=args.dataset.valid_split,
        valid_type=args.training.valid_type,
        one_hot=(args.strategy._name == "MCL"),
        num_cl=args.dataset.num_cl,
        transition_matrix=args.dataset.transition_matrix,
        noise=args.dataset.noise,
        seed=args.training.seed,
        uni_injection=args.strategy.uni_injection,
        data_augment=args.dataset.augment,
    )
    cl_data_module.prepare_data()
    cl_data_module.setup(stage="fit")
    train_loader, valid_loader, test_loader = cl_data_module.train_dataloader(), cl_data_module.val_dataloader(), cl_data_module.test_dataloader()

    input_dim, num_classes = cl_data_module.train_set.input_dim, cl_data_module.train_set.num_classes
    Q, class_priors = cl_data_module.get_distribution_info()
    # # Pretty print transition matrix Q
    # np.set_printoptions(precision=4, suppress=True)
    # print("Transition Matrix Q:")
    # print(Q.numpy())
    
    # # Save transition matrix to file
    # transition_matrix_file = f"{args.dataset._name}.txt"  # Can be changed as needed
    # np.savetxt(transition_matrix_file, Q.numpy(), fmt='%.6f', delimiter=' ')
    # print(f"Transition matrix saved to {transition_matrix_file}")


    print("Preparing Model......")

    pl.seed_everything(args.training.seed, workers=True)
    model = build_model(
        args.model._name,
        input_dim=input_dim,
        hidden_dim=args.model.hidden_dim,
        num_classes=num_classes,
    )

    strategy = build_strategy(
        args.strategy._name,
        model=model,
        valid_type=args.training.valid_type,
        num_classes=num_classes,
        type=args.strategy.type,
        lr=args.optimizer.lr,
        Q=Q,
        class_priors=class_priors,
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.training.output_dir)
    
    # Log hyperparameters to TensorBoard
    hparams = {
        "model": args.model._name,
        "dataset": args.dataset._name,
        "strategy": args.strategy._name,
        "strategy_type": args.strategy.type,
        "lr": args.optimizer.lr,
        "batch_size": args.training.batch_size,
        "num_cl": args.dataset.num_cl,
        "valid_type": args.training.valid_type,
        "eval_epoch": args.training.eval_epoch,
        "max_epochs": args.training.epoch,
        "seed": args.training.seed,
        "hidden_dim": args.model.hidden_dim,
        "data_augment": args.dataset.augment,
        "transition_matrix": args.dataset.transition_matrix,
        "noise": args.dataset.noise,
        "valid_split": args.dataset.valid_split,
    }
    
    # Initialize callbacks - will set log_dir after trainer creation
    test_callback = TestEpochCallback(test_loader)
    metrics_logger = MetricsLogger(args.training.output_dir)  # Temporary, will update
    summary_logger = SummaryLogger(args.training.output_dir, hparams)  # Temporary, will update
    
    callbacks = [test_callback, metrics_logger, summary_logger]
    
    if args.training.save_checkpoints:
        checkpoint_callback_best = ModelCheckpoint(
            monitor=f"Valid_{args.training.valid_type}",
            dirpath=args.training.output_dir,
            filename=f"{{epoch}}-{{Valid_{args.training.valid_type}:.2f}}",
            save_top_k=1,
            mode="max" if args.training.valid_type == "Accuracy" else "min",
            every_n_epochs=args.training.eval_epoch,
        )
        # checkpoint_callback_last = ModelCheckpoint(
        #     monitor=f"step",
        #     dirpath=args.training.output_dir,
        #     filename="{epoch}-{step}",
        #     save_top_k=1,
        #     mode="max",
        #     every_n_epochs=args.training.eval_epoch,
        # )
        # callbacks.extend([checkpoint_callback_best, checkpoint_callback_last])
        callbacks.append(checkpoint_callback_best)

    print("Start Training......")
    trainer = pl.Trainer(
        max_epochs=args.training.epoch,
        accelerator="gpu",
        devices=[args.training.gpu],
        logger=tb_logger,
        log_every_n_steps=args.training.log_step,
        check_val_every_n_epoch=args.training.eval_epoch,
        callbacks=callbacks,
        enable_checkpointing=args.training.save_checkpoints,
    )
    
    # Update logger paths to use TensorBoard logger's directory (same as hparams.yaml)
    log_dir = trainer.logger.log_dir
    metrics_logger.log_dir = log_dir
    metrics_logger.log_file = os.path.join(log_dir, "metrics.csv")
    summary_logger.log_dir = log_dir
    summary_logger.summary_file = os.path.join(log_dir, "summary.json")
    
    # Log hyperparameters with placeholder metrics (will be updated during training)
    trainer.logger.log_hyperparams(hparams, {"hp/best_test_acc": 0, "hp/best_valid_acc": 0})
    
    if args.training.do_train:
        with open(f"{args.training.output_dir}/config.yaml", "w") as f:
            OmegaConf.save(args, f)
        trainer.fit(
            strategy,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
        )
    if args.training.do_predict:
        if args.training.do_train:
            trainer.test(
                dataloaders=test_loader,  
                ckpt_path="best"
            )
        else:
            trainer.test(
                strategy, 
                dataloaders=test_loader, 
                ckpt_path=args.training.model_path
            )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config_file",
        dest="training.config_file",
        type=str,
        default="libcll/configs/base.yaml",
    )
    parser.add_argument("--model", dest="model._name", type=str, default="ResNet18")
    parser.add_argument(
        "--model_path", dest="training.model_path", type=str, default=None
    )
    parser.add_argument("--dataset", dest="dataset._name", type=str, default="mnist")
    parser.add_argument(
        "--valid_type", dest="training.valid_type", type=str, default="Accuracy"
    )
    parser.add_argument("--num_cl", dest="dataset.num_cl", type=int, default=1)
    parser.add_argument(
        "--valid_split", dest="dataset.valid_split", type=float, default=0.1
    )
    parser.add_argument(
        "--eval_epoch", dest="training.eval_epoch", type=int, default=10
    )
    parser.add_argument(
        "--output_dir", dest="training.output_dir", type=str, default=None
    )
    parser.add_argument(
        "--batch_size", dest="training.batch_size", type=int, default=512
    )
    parser.add_argument("--hidden_dim", dest="model.hidden_dim", type=int, default=500)
    parser.add_argument("--epoch", dest="training.epoch", type=int, default=300)
    parser.add_argument("--do_train", dest="training.do_train", action="store_true")
    parser.add_argument("--do_predict", dest="training.do_predict", action="store_true")
    parser.add_argument("--strategy", dest="strategy._name", type=str, default="SCL")
    parser.add_argument("--type", dest="strategy.type", type=str, default=None)
    parser.add_argument("--lr", dest="optimizer.lr", type=float, default=1e-3)
    parser.add_argument("--augment", dest="dataset.augment", type=str, default="flipflop",
                        help="Data augmentation type: flipflop (default), autoaug, randaug, cutout")
    parser.add_argument(
        "--transition_matrix",
        dest="dataset.transition_matrix",
        type=str,
        default="uniform",
    )
    parser.add_argument("--seed", dest="training.seed", type=int, default=1126)
    parser.add_argument("--log_step", dest="training.log_step", type=int, default=50)
    parser.add_argument("--noise", dest="dataset.noise", type=float, default=0.1)
    parser.add_argument("--gpu", dest="training.gpu", type=int, default=0)
    parser.add_argument("--save_checkpoints", dest="training.save_checkpoints", action="store_true", default=False)
    parser.add_argument("--uni-injection", dest="strategy.uni_injection", action="store_true", default=False)
    args = parser.parse_args()
    args = parse_config(
        parser, getattr(args, "training.config_file"), args=sys.argv[1:]
    )

    return args


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.training.output_dir, exist_ok=True)
    main(args)