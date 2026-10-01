"""Download model weights and tokenizer to the HF cache.

Run this on a Quest LOGIN node, which has internet access. Compute nodes
generally do not, so the download must happen before the job starts.

    export HF_TOKEN=hf_...            # needs Llama-3.1 license accepted
    python scripts/prefetch_model.py  # defaults to meta-llama/Llama-3.1-8B
"""

import argparse
import os

from huggingface_hub import snapshot_download


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", nargs="?", default="meta-llama/Llama-3.1-8B")
    args = p.parse_args()

    cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"HF_HOME={cache}")
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        print("WARNING: no HF_TOKEN set; gated repos such as Llama-3.1 will 401.")

    path = snapshot_download(
        args.model,
        allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
    )
    print(f"Downloaded {args.model} to {path}")


if __name__ == "__main__":
    main()
