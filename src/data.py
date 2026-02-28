"""Data module: Tokenizer, Datasets, Dataloaders.

Data is configured in config/config.yaml under 'data':
  - root: path to LibriSpeech-style dir (e.g. ./hw4_data/hw4p2_data)
  - train_partition, val_partition, test_partition: subdir names (train-clean-100, dev-clean, test-clean)
  - batch_size, NUM_WORKERS, norm, specaug, etc.

Override data root at runtime: main.py --data-root /path/to/your/data
Actual loading (fbank, text) is done by hw4lib.data.ASRDataset.
"""

import gc
from torch.utils.data import DataLoader

# Depends on hw4lib (add transformer-from-scratch to PYTHONPATH)
from hw4lib.data import H4Tokenizer, ASRDataset, verify_dataloader


def create_tokenizer(config: dict):
    """Create Tokenizer."""
    return H4Tokenizer(
        token_map=config["tokenization"]["token_map"],
        token_type=config["tokenization"]["token_type"],
    )


def create_datasets(config: dict, tokenizer):
    """
    Create train, validation, and test datasets.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = ASRDataset(
        partition=config["data"]["train_partition"],
        config=config["data"],
        tokenizer=tokenizer,
        isTrainPartition=True,
        global_stats=None,
    )

    global_stats = None
    if config["data"]["norm"] == "global_mvn":
        global_stats = (train_dataset.global_mean, train_dataset.global_std)

    val_dataset = ASRDataset(
        partition=config["data"]["val_partition"],
        config=config["data"],
        tokenizer=tokenizer,
        isTrainPartition=False,
        global_stats=global_stats,
    )

    test_dataset = ASRDataset(
        partition=config["data"]["test_partition"],
        config=config["data"],
        tokenizer=tokenizer,
        isTrainPartition=False,
        global_stats=global_stats,
    )

    gc.collect()
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    config: dict,
    device: str = "cuda",
):
    """Create DataLoaders."""
    num_workers = config["data"]["NUM_WORKERS"] if device == "cuda" else 0
    batch_size = config["data"]["batch_size"]

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=test_dataset.collate_fn,
    )

    gc.collect()
    return train_loader, val_loader, test_loader


def compute_max_lengths(train_dataset, val_dataset, test_dataset):
    """Compute max feature length and max transcript length."""
    max_feat_len = max(
        train_dataset.feat_max_len,
        val_dataset.feat_max_len,
        test_dataset.feat_max_len,
    )
    max_transcript_len = max(
        train_dataset.text_max_len,
        val_dataset.text_max_len,
        test_dataset.text_max_len,
    )
    max_len = max(max_feat_len, max_transcript_len)
    return max_feat_len, max_transcript_len, max_len
