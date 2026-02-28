"""Model loading module."""

from hw4lib.model import EncoderDecoderTransformer


def load_model(
    config: dict,
    max_len: int,
    num_classes: int,
    decoder_checkpoint_path: str = None,
):
    """
    Load Encoder-Decoder Transformer model.

    Supports initialization from pretrained Decoder checkpoint (for P2).

    Args:
        config: Config dictionary.
        max_len: Maximum sequence length.
        num_classes: Vocabulary size.
        decoder_checkpoint_path: P1 decoder checkpoint path; if None, random init.

    Returns:
        model, param_info
    """
    model_config = config["model"].copy()
    model_config.update({
        "max_len": max_len,
        "num_classes": num_classes,
    })

    if decoder_checkpoint_path:
        model, param_info = EncoderDecoderTransformer.from_pretrained_decoder(
            decoder_checkpoint_path=decoder_checkpoint_path,
            config=model_config,
        )
    else:
        model = EncoderDecoderTransformer(**model_config)
        param_info = {}

    return model, param_info
