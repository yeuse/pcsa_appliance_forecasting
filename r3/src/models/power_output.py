"""Explicit output maps. Legacy runs retain their original hard cap."""
import torch

def cap_amplitude(amplitude, cap, mode="hard"):
    if mode == "hard":
        return torch.minimum(amplitude, cap)
    if mode != "smooth":
        raise ValueError(f"Unknown power cap mode: {mode}")
    if not bool(torch.isfinite(cap).all()) or bool((cap <= 0).any()):
        raise ValueError("Smooth caps must be finite and positive")
    # FP32 avoids half-precision saturation. Polynomial derivative, no STE.
    a, c = amplitude.float(), cap.float()
    return c * (a / torch.hypot(a, c))
