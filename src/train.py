"""Training pipeline."""

from hw4lib.trainers import ASRTrainer, ProgressiveTrainer
from hw4lib.utils import create_optimizer, create_scheduler


def create_asr_trainer(model, tokenizer, config, run_name: str, device: str):
    """Create ASR Trainer (single-stage training)."""
    return ASRTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        run_name=run_name,
        config_file="config.yaml",
        device=device,
    )


def create_progressive_trainer(model, tokenizer, config, run_name: str, device: str):
    """Create Progressive Trainer (multi-stage progressive training)."""
    return ProgressiveTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        run_name=run_name,
        config_file="config.yaml",
        device=device,
    )


def setup_optimizer_and_scheduler(trainer, config, train_loader):
    """Setup optimizer and learning rate scheduler."""
    trainer.optimizer = create_optimizer(
        model=trainer.model,
        opt_config=config["optimizer"],
    )
    trainer.scheduler = create_scheduler(
        optimizer=trainer.optimizer,
        scheduler_config=config["scheduler"],
        train_loader=train_loader,
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
    )


# Progressive training stage config (extracted from notebook)
DEFAULT_PROGRESSIVE_STAGES = [
    {
        "name": "Stage 1: warmup encoder (small)",
        "epochs": 3,
        "encoder_active_layers": [0, 1],
        "decoder_active_layers": [0],
        "encoder_freeze": [False, False],
        "decoder_freeze": [True],
        "dropout": 0.0,
        "label_smoothing": 0.0,
        "data_subset": 0.3,
    },
    {
        "name": "Stage 2: full encoder, frozen decoder",
        "epochs": 5,
        "encoder_active_layers": [0, 1, 2, 3],
        "decoder_active_layers": [0, 1, 2],
        "encoder_freeze": [True, False, False, False],
        "decoder_freeze": [True, True, True],
        "dropout": 0.05,
        "label_smoothing": 0.05,
        "data_subset": 0.5,
    },
    {
        "name": "Stage 3: train top decoder layers",
        "epochs": 6,
        "encoder_active_layers": [0, 1, 2, 3],
        "decoder_active_layers": [0, 1, 2, 3, 4, 5],
        "encoder_freeze": [False, False, False, False],
        "decoder_freeze": [True, True, True, False, False, False],
        "dropout": 0.1,
        "label_smoothing": 0.1,
        "data_subset": 1.0,
    },
    {
        "name": "Stage 4: finetune all layers",
        "epochs": 26,
        "encoder_active_layers": [0, 1, 2, 3],
        "decoder_active_layers": [0, 1, 2, 3, 4, 5],
        "encoder_freeze": [False, False, False, False],
        "decoder_freeze": [False, False, False, False, False, False],
        "dropout": 0.1,
        "label_smoothing": 0.1,
        "data_subset": 1.0,
    },
]
