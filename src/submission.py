"""Submission: model metadata generation, Kaggle submission."""

import json
import os
import sys
import datetime


def is_colab():
    return "google.colab" in sys.modules and "COLAB_GPU" in os.environ


def is_kaggle():
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or "KAGGLE_URL_BASE" in os.environ


def generate_model_metadata(model, output_dir: str = "."):
    """
    Generate model_metadata.json file (for Autolab submission).

    Args:
        model: PyTorch model.
        output_dir: Output directory.

    Returns:
        Path to the generated file.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    json_filename = os.path.join(output_dir, f"model_metadata_{timestamp}.json")

    output_json = {
        "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "model_architecture": str(model),
    }

    with open(json_filename, "w") as f:
        json.dump(output_json, f, indent=2)

    if is_colab():
        from google.colab import files
        print(f"OK: Saved as {json_filename}. Downloading in Colab...")
        files.download(json_filename)
    elif is_kaggle():
        from IPython.display import FileLink, display
        print("#" * 100)
        print(f"OK: Your submission file `{json_filename}` has been generated.")
        print("TODO: Click the link below.")
        print("1. The file will open in a new tab.")
        print("2. Right-click anywhere in the new tab and select 'Save As...'")
        print("3. Save the file to your computer with the `.json` extension.")
        print("You MUST submit this file to Autolab if this is your best submission.")
        print("#" * 100 + "\n")
        display(FileLink(json_filename))
    else:
        print(f"OK: Model metadata saved to: '{json_filename}'")
        print("REQUIRED to submit to Autolab if these are the best model weights.")

    return json_filename


def save_results_csv(results_df, path: str = "results.csv"):
    """Save inference results to CSV."""
    results_df.to_csv(path, index=False)
    print(f"Results saved to {path}")
