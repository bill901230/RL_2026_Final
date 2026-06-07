# pyright: reportAttributeAccessIssue=false, reportMissingImports=false
# pyright: reportMissingTypeArgument=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownParameterType=false
# pyright: reportUnknownVariableType=false, reportUntypedClassDecorator=false

import os
import pprint


ProbeResult = dict[str, str | bool | int | None]

import ray


def main() -> None:
    ray.init(
        _temp_dir=os.environ["RAY_TMPDIR"],
        include_dashboard=False,
        object_store_memory=8 * 1024**3,
        num_cpus=8,
    )

    @ray.remote(num_gpus=1, num_cpus=0)
    class GpuProbe:
        def probe(self) -> ProbeResult:
            import os

            import torch

            return {
                "pid": os.getpid(),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None,
            }

    actors = [GpuProbe.remote() for _ in range(2)]
    results = ray.get([actor.probe.remote() for actor in actors], timeout=30)
    pprint.pp(results)

    assert len(results) == 2, results
    assert all(result["cuda_available"] for result in results), results
    visible = [result["CUDA_VISIBLE_DEVICES"] for result in results]
    assert len(set(visible)) == 2, visible
    print("RAY_GPU_SMOKE_OK")
    ray.shutdown()


if __name__ == "__main__":
    main()
