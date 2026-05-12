from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

try:
    import psutil
except Exception:  # pragma: no cover - only used before dependencies are installed
    psutil = None


CPU_FALLBACK_MESSAGE = "CPU fallback only, not valid for final AMD GPU requirement"
BACKEND_NOTE = "PyTorch CUDA interface; NVIDIA CUDA in current WSL, AMD ROCm/HIP on final AMD workstation"


@dataclass
class HardwareInfo:
    actual_device: str
    uses_accelerator: bool
    torch_version: str
    torch_cuda_available: bool
    device_name: str
    backend_note: str
    cpu_count: int
    memory_total_mb: float
    npu_available: bool
    npu_note: str
    warning: str | None = None

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
        warning = None
    else:
        actual_device = "cpu"
        device_name = "CPU"
        warning = CPU_FALLBACK_MESSAGE

    cpu_count = psutil.cpu_count(logical=True) if psutil else 0
    memory_total_mb = round(psutil.virtual_memory().total / 1024 / 1024, 2) if psutil else 0.0
    return HardwareInfo(
        actual_device=actual_device,
        uses_accelerator=cuda_available,
        torch_version=torch.__version__,
        torch_cuda_available=cuda_available,
        device_name=device_name,
        backend_note=BACKEND_NOTE,
        cpu_count=cpu_count or 0,
        memory_total_mb=memory_total_mb,
        npu_available=False,
        npu_note="NPU is not used in this MVP; future Ryzen AI adaptation can export ONNX and run via Ryzen AI/Vitis AI stack.",
        warning=warning,
    )
