import time
from typing import Callable, Literal

from rich import print
import torch


def calculate_cpu_runtime(fct: Callable) -> float:
    """Determines the CPU runtime of any function.

    Does not take GPU runtime into account.

    Args:
        fct (Callable): The training or prediction function to determine the runtime for.

    Returns:
        float: The runtime in seconds.
    """
    start_time = time.time()
    fct()
    elapsed_time = (time.time() - start_time) * 1000

    return elapsed_time


def calculate_runtime_pytorch(fct: Callable, device: Literal["cpu", "cuda"], profiler: bool = False, **kwargs):
    """Determines the runtime of PyTorch functions.

    Note the difference between self cpu/cuda time and cpu/cuda time - operators can call other operators,
    self cpu/cuda time excludes time spent in children operator calls, while total cpu/cuda time includes it.

    Args:
        fct (Callable): The training or prediction function to determine the runtime for.
        device (str): The device to profile. One of "cpu" or "cuda" (GPUs).
        profiler (bool, optional): Whether to profile using the PyTorch profiler. Defaults to False.
    """
    cuda_enabled = True if device == "cuda" else False

    if profiler:
        with torch.autograd.profiler.profile(use_cuda=cuda_enabled, **kwargs) as prof:
            fct()

        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        print(f"[bold blue]Total average CPU runtime: {(prof.total_average().cpu_time / 1000) :.3f} seconds")
        print(f"[bold blue]Total average GPU runtime: {(prof.total_average().cuda_time / 1000):.3f} seconds")
    elif not cuda_enabled:  # CPU without profiler
        print(f"[bold blue]Total average CPU runtime: {(calculate_cpu_runtime(fct) / 1000):.3f} seconds")
    else:  # GPU without profiler
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fct()
        end.record()

        torch.cuda.synchronize()
        print(f"[bold blue]Total average GPU runtime: {start.elapsed_time(end) / 1000:.3f}")
