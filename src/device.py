"""Utilities for selecting the CPU or CUDA execution device."""

def get_device(requested: str = "auto") -> str:
    """Resolve a user-requested device string to an available runtime target."""
    requested = (requested or "auto").lower().strip()

    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be one of: auto, cpu, cuda")

    if requested == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError as exc:
        if requested == "cuda":
            raise RuntimeError("PyTorch is not installed, so CUDA cannot be used.") from exc
        return "cpu"

    has_cuda = torch.cuda.is_available()

    if requested == "cuda" and not has_cuda:
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() returned False.")

    if requested == "auto":
        return "cuda" if has_cuda else "cpu"

    return "cuda"
