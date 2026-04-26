"""CLI helper for inspecting the configured compute device."""

import argparse
from device import get_device

def main() -> None:
    """Parse a requested device and print the resolved runtime details."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    selected = get_device(args.device)
    print(f"Requested device: {args.device}")
    print(f"Selected device: {selected}")

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch is not installed.")

if __name__ == "__main__":
    main()
