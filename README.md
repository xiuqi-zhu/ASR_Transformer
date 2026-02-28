# ASR Transformer - Encoder-Decoder Automatic Speech Recognition

This project is extracted from **HW4P2 (Automatic Speech Recognition with an Encoder-Decoder Transformer)** Jupyter notebook into a standalone Python project for version control and deployment.

**This is a separate repository.** It depends on [transformer-from-scratch](https://github.com/xiuqi-zhu/transformer-from-scratch) (hw4lib, mytorch, tests) via `PYTHONPATH`.

## Project Structure

```
asr_transformer/
├── config/
│   ├── __init__.py
│   └── config.yaml          # Main config file
├── src/
│   ├── __init__.py
│   ├── config_loader.py     # Config loading
│   ├── data.py              # Tokenizer, datasets, DataLoader
│   ├── model.py             # Model loading
│   ├── train.py             # Training pipeline (ASRTrainer / ProgressiveTrainer)
│   ├── inference.py         # Inference pipeline
│   └── submission.py        # Submission (model_metadata, results.csv)
├── main.py                  # Main entry point
├── run.sh                   # Helper script (Linux/macOS)
├── run.ps1                  # Helper script (Windows)
├── requirements.txt         # Python dependencies
└── README.md
```

## Prerequisites

This repository depends on **transformer-from-scratch** (hw4lib, mytorch, tests). Clone it and add its root to `PYTHONPATH`:

```bash
# Clone the base repo
git clone https://github.com/xiuqi-zhu/transformer-from-scratch.git

# Clone this repo (e.g. alongside transformer-from-scratch)
git clone <your-asr-transformer-repo-url>
cd asr_transformer
```

## Installation

```bash
# Create virtual env (recommended)
conda create -n hw4 python=3.12
conda activate hw4

# Install dependencies
pip install -r requirements.txt
pip install -r ../transformer-from-scratch/requirements.txt  # if not already installed
```

## Running (PYTHONPATH)

**Option A – Use helper script (recommended):**

```bash
# Linux / macOS
./run.sh python main.py --mode train --decoder-checkpoint path/to/lm.pth

# Windows (PowerShell)
.\run.ps1 python main.py --mode train --decoder-checkpoint path/to/lm.pth
```

By default, the script expects `transformer-from-scratch` in the parent directory. Override with:
- Linux/macOS: `export TRANSFORMER_SCRATCH_DIR=/path/to/transformer-from-scratch`
- Windows: `$env:TRANSFORMER_SCRATCH_DIR = 'C:\path\to\transformer-from-scratch'`

**Option B – Set PYTHONPATH manually:**

```bash
# Linux / macOS
export PYTHONPATH=/path/to/transformer-from-scratch:$PYTHONPATH
python main.py --mode train --decoder-checkpoint path/to/lm.pth

# Windows (PowerShell)
$env:PYTHONPATH = "C:\path\to\transformer-from-scratch;" + $env:PYTHONPATH
python main.py --mode train --decoder-checkpoint path/to/lm.pth
```

**Option C – One-liner:**

```bash
PYTHONPATH=/path/to/transformer-from-scratch python main.py --mode train --decoder-checkpoint path/to/lm.pth
```

## Configuration

Edit `config/config.yaml`:

- `data.root`: Data root path (LibriSpeech format)
- `tokenization.token_type`: Tokenization type (char / 1k / 5k / 10k)
- `model`: Model hyperparameters
- `training`: Training settings
- `optimizer` / `scheduler`: Optimizer and LR schedule

## Usage

### 1. Training

```bash
# Single-stage training (from random/pretrained decoder init)
python main.py --mode train --decoder-checkpoint path/to/lm_p1_decoder_for_p2.pth --run-name hw4p2_run --epochs 30

# Progressive multi-stage training
python main.py --mode train --decoder-checkpoint path/to/lm_p1_decoder_for_p2.pth --progressive --run-name hw4p2_pro
```

### 2. Inference

```bash
# Greedy / Beam Search inference
python main.py --mode inference --trainer-checkpoint path/to/checkpoint-best-metric-model.pth --output results.csv

# Shallow Fusion
python main.py --mode inference --trainer-checkpoint path/to/checkpoint.pth --shallow-fusion --lm-checkpoint path/to/lm_p1_decoder_for_p2.pth
```

### 3. Full Pipeline (train + inference)

```bash
python main.py --mode full --decoder-checkpoint path/to/lm_p1_decoder_for_p2.pth --run-name hw4p2_full --epochs 30
```

### 4. Override Data Path

```bash
python main.py --mode train --data-root /path/to/hw4_data/hw4p2_data
```

## CLI Arguments

| Argument | Description |
|----------|-------------|
| `--mode` | `train` / `inference` / `full` |
| `--config` | Config file path |
| `--decoder-checkpoint` | P1 Decoder checkpoint (init Encoder-Decoder) |
| `--trainer-checkpoint` | Trainer checkpoint (resume or inference) |
| `--run-name` | Wandb run name |
| `--epochs` | Epochs for single-stage training |
| `--progressive` | Use progressive training |
| `--shallow-fusion` | Enable Shallow Fusion for inference |
| `--lm-checkpoint` | LM checkpoint path (Shallow Fusion) |
| `--data-root` | Override config data.root |
| `--output` | Output CSV path for inference results |

## Data

Data must follow LibriSpeech format:

```
hw4_data/hw4p2_data/
├── train-clean-100/
├── dev-clean/
└── test-clean/
```

Obtain via Kaggle API or course-provided links.

## License

Based on HW4P2 assignment framework, for educational use only.
