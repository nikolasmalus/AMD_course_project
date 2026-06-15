from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

try:
    import psutil
except Exception:  # pragma: no cover - only used before dependencies are installed
    psutil = None


@dataclass
class HardwareInfo:
    actual_device: str
    torch_version: str
    device_name: str
    cpu_count: int
    memory_total_mb: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_hardware() -> HardwareInfo:
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        actual_device = "cuda"
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = "Unknown PyTorch CUDA-compatible device"
    else:
        actual_device = "cpu"
        device_name = "CPU"

    cpu_count = psutil.cpu_count(logical=True) if psutil else 0
    memory_total_mb = round(psutil.virtual_memory().total / 1024 / 1024, 2) if psutil else 0.0
    return HardwareInfo(
        actual_device=actual_device,
        torch_version=torch.__version__,
        device_name=device_name,
        cpu_count=cpu_count or 0,
        memory_total_mb=memory_total_mb,
    )
