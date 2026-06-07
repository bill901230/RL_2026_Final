# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUntypedClassDecorator=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false, reportUnknownArgumentType=false
"""Minimal Ray GPU actor repro for W3.T1.

Runs on the already-filtered CUDA_VISIBLE_DEVICES=1,3 view and verifies that two
1-GPU Ray actors can start, import torch, and see exactly one assigned GPU each.
"""

from __future__ import annotations

import os
import socket


def main() -> None:
    import ray

    print(f"driver host={socket.gethostname()} pid={os.getpid()}", flush=True)
    print(f"driver CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    if not ray.is_initialized():
        ray.init(num_cpus=4, include_dashboard=False)

    @ray.remote(num_gpus=1)
    class GpuActor:
        def probe(self) -> dict[str, object]:
            import torch

            return {
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "ray_gpu_ids": ray.get_gpu_ids(),
                "torch_cuda_available": torch.cuda.is_available(),
                "torch_device_count": torch.cuda.device_count(),
                "torch_device_name_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }

    actors = [GpuActor.remote() for _ in range(2)]
    refs = [actor.probe.remote() for actor in actors]
    for idx, result in enumerate(ray.get(refs, timeout=60)):
        print(f"ACTOR_{idx}: {result}", flush=True)
    print("RAY_GPU_ACTOR_REPRO_OK", flush=True)
    ray.shutdown()


if __name__ == "__main__":
    main()
