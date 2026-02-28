#!/usr/bin/env python3
"""
ASR Transformer - Main entry point.

Converts HW4P2 Jupyter notebook into an executable Python project.
Requires hw4lib (place handout directory in project root or PYTHONPATH).

Usage:
    python main.py --mode train           # Train
    python main.py --mode inference       # Inference
    python main.py --mode full            # Full pipeline (train + inference)
"""

import argparse
import os
import sys

# Add project root to path for hw4lib
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from src.config_loader import load_config
from src.data import (
    create_tokenizer,
    create_datasets,
    create_dataloaders,
    compute_max_lengths,
    verify_dataloader,
)
from src.model import load_model
from src.train import (
    create_asr_trainer,
    create_progressive_trainer,
    setup_optimizer_and_scheduler,
    DEFAULT_PROGRESSIVE_STAGES,
)
from src.inference import run_recognition, load_lm_for_shallow_fusion, save_results_csv


def parse_args():
    parser = argparse.ArgumentParser(description="ASR Transformer - HW4P2")
    parser.add_argument(
        "--mode",
        choices=["train", "inference", "full"],
        default="full",
        help="Run mode: train / inference / full",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config file path (default: config/config.yaml)",
    )
    parser.add_argument(
        "--decoder-checkpoint",
        default=None,
        help="P1 Decoder checkpoint path (to init Encoder-Decoder)",
    )
    parser.add_argument(
        "--trainer-checkpoint",
        default=None,
        help="Trainer checkpoint path (for resuming or inference)",
    )
    parser.add_argument(
        "--run-name",
        default="hw4p2_run",
        help="Wandb run name",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs for single-stage training",
    )
    parser.add_argument(
        "--progressive",
        action="store_true",
        help="Use progressive training (ProgressiveTrainer)",
    )
    parser.add_argument(
        "--shallow-fusion",
        action="store_true",
        help="Use Shallow Fusion during inference (requires --lm-checkpoint)",
    )
    parser.add_argument(
        "--lm-checkpoint",
        default=None,
        help="LM checkpoint path (for Shallow Fusion)",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override config data.root path",
    )
    parser.add_argument(
        "--output",
        default="results.csv",
        help="Output CSV path for inference results",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure working directory is project root (contains config/, hw4lib/, etc.)
    os.chdir(PROJECT_ROOT)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load config
    config = load_config(args.config)
    if args.data_root:
        config["data"]["root"] = args.data_root

    # 1. Create tokenizer and datasets
    print("Creating tokenizer and datasets...")
    tokenizer = create_tokenizer(config)
    train_dataset, val_dataset, test_dataset = create_datasets(config, tokenizer)

    # 2. Create Dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, config, device
    )

    # 3. Compute max lengths
    max_feat_len, max_transcript_len, max_len = compute_max_lengths(
        train_dataset, val_dataset, test_dataset
    )
    print(f"Max Feature Length: {max_feat_len}, Max Transcript Length: {max_transcript_len}")

    # 4. Load model
    print("Loading model...")
    model, _ = load_model(
        config=config,
        max_len=max_len,
        num_classes=tokenizer.vocab_size,
        decoder_checkpoint_path=args.decoder_checkpoint,
    )
    model = model.to(device)

    # 5. Create trainer
    trainer = create_asr_trainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        run_name=args.run_name,
        device=device,
    )

    # Load checkpoint (if specified)
    if args.trainer_checkpoint:
        trainer.load_checkpoint(args.trainer_checkpoint)

    # 6. Train (if needed)
    if args.mode in ("train", "full"):
        setup_optimizer_and_scheduler(trainer, config, train_loader)

        if args.progressive:
            progressive_trainer = create_progressive_trainer(
                model=model,
                tokenizer=tokenizer,
                config=config,
                run_name=args.run_name,
                device=device,
            )
            progressive_trainer.optimizer = trainer.optimizer
            progressive_trainer.scheduler = trainer.scheduler
            progressive_trainer.progressive_train(
                train_loader, val_loader, DEFAULT_PROGRESSIVE_STAGES
            )
            trainer = progressive_trainer
        else:
            trainer.train(train_loader, val_loader, epochs=args.epochs)

    # 7. Inference (if needed)
    if args.mode in ("inference", "full"):
        lm_model = None
        lm_weight = 0.2
        if args.shallow_fusion and args.lm_checkpoint:
            lm_model = load_lm_for_shallow_fusion(
                args.lm_checkpoint, tokenizer, device
            )
            config_name = "shallow_fusion"
            beam_width = 6
        else:
            config_name = "test"
            beam_width = 5

        results_df = run_recognition(
            trainer=trainer,
            test_loader=test_loader,
            max_transcript_len=max_transcript_len,
            config_name=config_name,
            lm_model=lm_model,
            lm_weight=lm_weight,
            beam_width=beam_width,
        )

        save_results_csv(results_df, args.output)
        trainer.cleanup()

    print("Done.")


if __name__ == "__main__":
    main()
