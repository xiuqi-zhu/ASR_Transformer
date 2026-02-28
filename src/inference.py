"""Inference pipeline."""

import pandas as pd


def save_results_csv(results_df, path: str = "results.csv"):
    """Save inference results to CSV."""
    results_df.to_csv(path, index=False)
    print(f"Results saved to {path}")


def run_recognition(
    trainer,
    test_loader,
    max_transcript_len: int,
    config_name: str = "test",
    lm_model=None,
    lm_weight: float = 0.2,
    beam_width: int = 5,
):
    """
    Run ASR inference.

    Args:
        trainer: ASRTrainer instance.
        test_loader: Test DataLoader.
        max_transcript_len: Maximum transcript length.
        config_name: Config name (for saving results).
        lm_model: Language model for shallow fusion; None to disable.
        lm_weight: LM weight (only used when lm_model is not None).
        beam_width: Beam search width.

    Returns:
        results_df: DataFrame with id and transcription columns.
    """
    recognition_config = {
        "num_batches": None,
        "temperature": 1.0,
        "repeat_penalty": 1.0,
        "lm_weight": lm_weight if lm_model else None,
        "lm_model": lm_model,
        "beam_width": beam_width,
    }

    print(f"Evaluating with {config_name} config")
    results = trainer.recognize(
        test_loader,
        recognition_config,
        config_name=config_name,
        max_length=max_transcript_len,
    )

    generated = [r["generated"] for r in results]
    results_df = pd.DataFrame({
        "id": range(len(generated)),
        "transcription": generated,
    })

    return results_df


def load_lm_for_shallow_fusion(
    lm_checkpoint_path: str,
    tokenizer,
    device,
    num_layers: int = 6,
    d_model: int = 384,
    num_heads: int = 6,
    d_ff: int = 1536,
    dropout: float = 0.1,
    max_len: int = 120,
):
    """
    Load Decoder-only LM for Shallow Fusion.

    Must use the same config as P1 to construct decoder-only LM.
    """
    import torch
    from hw4lib.model.transformers import DecoderOnlyTransformer
    from hw4lib.model.sublayers import PositionalEncoding

    lm_model = DecoderOnlyTransformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        max_len=max_len,
        num_classes=tokenizer.vocab_size,
        weight_tying=True,
    )

    state = torch.load(lm_checkpoint_path, map_location="cpu", weights_only=False)
    lm_model.load_state_dict(state["model_state_dict"])

    lm_model.positional_encoding = PositionalEncoding(
        d_model=d_model,
        max_len=2048,
    ).to(device)
    lm_model = lm_model.to(device)
    lm_model.eval()

    print("Loaded LM for shallow fusion!")
    return lm_model
