"""Detects and manages available compute devices for PyTorch models."""
import torch


class DeviceDetector:
    """Detects the best available compute device for PyTorch operations."""

    def __init__(self):
        self._device = self._detect_device()

    def _detect_device(self):
        """Detects the best available device: MPS (macOS) > CUDA > CPU."""
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def get_device(self):
        """Returns the detected device string."""
        return self._device

    def get_device_info(self):
        """Returns detailed information about the detected device."""
        device_info = {
            "device": self._device,
            "is_cuda": torch.cuda.is_available(),
            "is_mps": torch.backends.mps.is_available(),
            "is_cpu": self._device == "cpu"
        }

        if self._device == "cuda":
            device_info["cuda_version"] = torch.version.cuda
            device_info["device_count"] = torch.cuda.device_count()
            device_info["device_names"] = [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ]

        return device_info

