#!/usr/bin/env python
"""
Copyright (c) 2025  Alireza Hajebrahimi. All rights reserved.

This work is the original creation of Alireza Hajebrahimi and was developed as part of 
[Course Name and Code, e.g., "Advanced Deep Learning (CS699)"] at 
[Your University/Institution Name] in [Year].

Ownership & License
-------------------
This source code and the associated model architecture, weights, and documentation 
are proprietary to Alireza Hajebrahimi. No part of this work may be reproduced, 
distributed, modified, or used to create derivative works (including training new 
models on top of this architecture or weights) for commercial purposes without 
explicit written permission from the author.

Permitted Use
-------------
You may use, study, and modify this code only for non-commercial, academic, or 
personal research purposes provided that:

1. You retain this entire copyright and permission notice in all copies or 
   substantial portions of the work.
2. Any use, publication, or distribution of this work (or derivatives) includes 
   the following mandatory citation:

   @software{yourname_yourmodel_2025,
     author    = {Alireza Hajebrahimi},
     title     = {[Project/Model Name] -- A novel deep learning model},
     year      = {2025},
     note      = {Developed as part of [Course Name] at [University Name]},
     url       = {[GitHub/Link to your repo, if public]},
     version   = {1.0}
   }

   (or an equivalent citation in the format required by your venue)

3. You clearly indicate any modifications made to the original work.

NO WARRANTY
-----------
This work is provided "AS IS" without any warranty of any kind, either express or 
implied, including but not limited to merchantability, fitness for a particular 
purpose, or non-infringement.

Contact
-------
For licensing inquiries, commercial use, or any questions: 
[your.email@example.com]


BrainCLIP Training Script
=========================

Main training script for BrainCLIP model using PyTorch Lightning and WandB.

Usage:
    python train.py --config configs/config.yaml
    python train.py --data_dir data/hdf5_data_final --batch_size 64
    
Example with custom settings:
    python train.py \
        --data_dir data/hdf5_data_final \
        --batch_size 64 \
        --learning_rate 1e-4 \
        --max_epochs 65 \
        --wandb_project brainclip \
        --experiment_name baseline_v1
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.brain_encoder import BrainEncoderConfig
from models.text_encoder import TextEncoderConfig
from models.brainclip import BrainCLIPConfig, ProjectionConfig
from data.preprocessing import AugmentationConfig
from brainlight.module import BrainCLIPLightningModule
from brainlight.datamodule import BrainCLIPDataModule
from brainlight.callbacks import get_callbacks


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train BrainCLIP model')
    
    # Config file
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML file')
    
    # Data settings
    parser.add_argument('--data_dir', type=str, default='data/hdf5_data_final', help='Path to data directory')
    parser.add_argument('--max_seq_len', type=int, default=256, help='Maximum sequence length (time bins)')
    
    # Model settings
    parser.add_argument('--brain_hidden_dim', type=int, default=256, help='Brain encoder hidden dimension')
    parser.add_argument('--brain_num_temporal_layers', type=int, default=4, help='Number of temporal transformer layers')
    parser.add_argument('--brain_num_spatiotemporal_layers', type=int, default=4, help='Number of spatiotemporal transformer layers')
    parser.add_argument('--text_model', type=str, default='distilbert-base-uncased', help='Pretrained text model name')
    parser.add_argument('--freeze_text', action='store_true', default=True, help='Freeze text encoder')
    parser.add_argument('--no_freeze_text', action='store_false', dest='freeze_text', help='Do not freeze text encoder')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Shared embedding dimension')
    parser.add_argument('--temperature', type=float, default=0.1, help='Contrastive loss temperature')
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.03, help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=65, help='Maximum epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--precision', type=str, default='bf16-mixed', choices=['32', '16-mixed', 'bf16-mixed'], help='Training precision')
    parser.add_argument('--dev', action='store_true', help='Run in development mode (fast_dev_run)')
    
    # Regularization settings - BALANCED
    parser.add_argument('--label_smoothing', type=float, default=0.05, help='Label smoothing for contrastive loss')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for brain encoder')
    parser.add_argument('--attention_dropout', type=float, default=0.1, help='Attention dropout rate')
    parser.add_argument('--layer_drop_rate', type=float, default=0.05, help='Stochastic depth / layer dropout rate')
    
    # Augmentation settings - BALANCED
    parser.add_argument('--no_augmentation', action='store_true', help='Disable data augmentation')
    parser.add_argument('--time_mask_prob', type=float, default=0.5, help='Time masking probability')
    parser.add_argument('--channel_dropout_prob', type=float, default=0.2, help='Channel dropout probability')
    parser.add_argument('--noise_std', type=float, default=0.1, help='Noise standard deviation')
    
    # Logging settings
    parser.add_argument('--wandb_project', type=str, default='brainclip', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--no_wandb', action='store_true', help='Disable WandB logging')
    
    # Hardware settings
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Resume training
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(file_config: dict, args: argparse.Namespace) -> dict:
    """Merge file config with command line args (CLI takes precedence)."""
    config = file_config.copy()
    
    # Update with non-None CLI arguments
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            config[key] = value
    
    return config


def create_model_config(args) -> BrainCLIPConfig:
    """Create model configuration from arguments."""
    brain_config = BrainEncoderConfig(
        hidden_dim=args.brain_hidden_dim,
        num_temporal_layers=args.brain_num_temporal_layers,
        num_spatiotemporal_layers=args.brain_num_spatiotemporal_layers,
        dropout=getattr(args, 'dropout', 0.3),
        attention_dropout=getattr(args, 'attention_dropout', 0.2),
        layer_drop_rate=getattr(args, 'layer_drop_rate', 0.1),
    )
    
    text_config = TextEncoderConfig(
        model_name=args.text_model,
        freeze=args.freeze_text,
    )
    
    projection_config = ProjectionConfig(
        output_dim=args.embedding_dim,
        dropout=getattr(args, 'dropout', 0.2),  # Use same dropout for projection
    )
    
    return BrainCLIPConfig(
        brain_encoder=brain_config,
        text_encoder=text_config,
        projection=projection_config,
        temperature=args.temperature,
    )


def create_augmentation_config(args) -> AugmentationConfig:
    """Create augmentation configuration from arguments."""
    return AugmentationConfig(
        time_mask_enabled=not args.no_augmentation,
        time_mask_prob=args.time_mask_prob,
        channel_dropout_enabled=not args.no_augmentation,
        channel_dropout_prob=args.channel_dropout_prob,
        noise_enabled=not args.no_augmentation,
        noise_std=args.noise_std,
    )


def main():
    """Main training function."""
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
    args = parse_args()
    
    # Load config file if provided
    if args.config is not None:
        file_config = load_config(args.config)
        # Convert to namespace for easier access
        for key, value in file_config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    # Set random seed
    pl.seed_everything(args.seed, workers=True)
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f'brainclip_{timestamp}'
    
    print(f"\n{'='*60}")
    print("BrainCLIP Training")
    print(f"{'='*60}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Data: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Label smoothing: {getattr(args, 'label_smoothing', 0.1)}")
    print(f"Dropout: {getattr(args, 'dropout', 0.3)}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"{'='*60}\n")
    
    # Create model configuration
    model_config = create_model_config(args)
    
    # Create Lightning module
    module = BrainCLIPLightningModule(
        model_config=model_config,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        label_smoothing=getattr(args, 'label_smoothing', 0.1),
    )
    
    # Get tokenizer from model
    tokenizer = module.get_tokenizer()
    
    # Create augmentation config
    aug_config = create_augmentation_config(args)
    
    # Create DataModule
    datamodule = BrainCLIPDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
        tokenizer=tokenizer,
        augmentation_config=aug_config,
        use_augmentation=not args.no_augmentation,
    )
    
    # Create logger
    if args.no_wandb:
        logger = None
    else:
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.experiment_name,
            save_dir='wandb_logs',
            log_model=True,
        )
        
        # Log hyperparameters
        logger.log_hyperparams({
            'brain_hidden_dim': args.brain_hidden_dim,
            'brain_num_temporal_layers': args.brain_num_temporal_layers,
            'brain_num_spatiotemporal_layers': args.brain_num_spatiotemporal_layers,
            'text_model': args.text_model,
            'freeze_text': args.freeze_text,
            'embedding_dim': args.embedding_dim,
            'temperature': args.temperature,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'max_epochs': args.max_epochs,
            'warmup_epochs': args.warmup_epochs,
            'use_augmentation': not args.no_augmentation,
        })
    
    # Create callbacks
    callbacks = get_callbacks(
        project_name=args.wandb_project,
        checkpoint_dir=args.checkpoint_dir,
        log_samples=not args.no_wandb,
        monitor_gradients=not args.no_wandb,
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='auto',
        devices=args.gpus if torch.cuda.is_available() else 1,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
        enable_progress_bar=True,
        log_every_n_steps=50,
        val_check_interval=1.0,
        fast_dev_run=args.dev,
    )
    
    # Train
    print("Starting training...")
    trainer.fit(
        module,
        datamodule=datamodule,
        ckpt_path=args.resume_from,
        weights_only=False,  # Required for checkpoints with custom config classes
    )
    
    # Test
    print("\nRunning final evaluation...")
    trainer.test(module, datamodule=datamodule)
    
    # Save final model
    final_path = Path(args.checkpoint_dir) / 'final_model.ckpt'
    trainer.save_checkpoint(final_path)
    print(f"\nFinal model saved to: {final_path}")
    
    if logger is not None:
        # Log final model artifact
        logger.experiment.log_artifact(str(final_path), type='model')
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()