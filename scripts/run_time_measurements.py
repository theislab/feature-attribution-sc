import functools
from typing import Callable

import torch


def calculate_runtime(device: str, fct: Callable, profiler: bool = False, **kwargs):
    cuda_enabled = True if device == "cuda" else False

    if profiler:
        with torch.autograd.profiler.profile(use_cuda=cuda_enabled, **kwargs) as prof:
            fct()

        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        print(f"Total average CPU runtime: {prof.total_average().cpu_time}")
        print(f"Total average GPU runtime: {prof.total_average().gpu_time}")
    elif not cuda_enabled:  # CPU without profiler
        start_time = time.time()
        fct()
        elapsed_time = (time.time() - start_time) * 1000
        print(f"Total average CPU runtime: {elapsed_time}")
    else:  # GPU without profiler
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fct()
        end.record()

        torch.cuda.synchronize()
        print(f"Total average GPU runtime: {start.elapsed_time(end) / 1000}")


def calculate_runtime_dec(device: str, profiler: bool, **kwargs):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cuda_enabled = True if device == "cuda" else False

            if profiler:
                with torch.autograd.profiler.profile(use_cuda=cuda_enabled, **kwargs) as prof:
                    func(*args, **kwargs)

                print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                print(f"Total average CPU runtime: {prof.total_average().cpu_time}")
                print(f"Total average GPU runtime: {prof.total_average().gpu_time}")
            elif not cuda_enabled:  # CPU without profiler
                start_time = time.time()
                func(*args, **kwargs)
                elapsed_time = (time.time() - start_time) * 1000
                print(f"Total average CPU runtime: {elapsed_time}")
            else:  # GPU without profiler
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                func(*args, **kwargs)
                end.record()

                torch.cuda.synchronize()
                print(f"Total average GPU runtime: {start.elapsed_time(end) / 1000}")

        return wrapper

    return decorator
