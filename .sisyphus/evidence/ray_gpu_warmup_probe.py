# pyright: reportAttributeAccessIssue=false, reportMissingImports=false
# pyright: reportMissingTypeArgument=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownParameterType=false
# pyright: reportUnknownVariableType=false, reportUntypedClassDecorator=false

import os
import pprint

import ray


ProbeResult = dict[str, str | bool | int | None]


def main() -> None:
    try:
        ray.init(
            _temp_dir=os.environ["RAY_TMPDIR"],
            include_dashboard=False,
            object_store_memory=8 * 1024**3,
            num_cpus=8,
        )

        @ray.remote(num_gpus=1, num_cpus=0)
        class WarmTorchProbe:
            def __init__(self) -> None:
                import os

                import torch

                self.pid = os.getpid()
                self.visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                self.cuda_available = torch.cuda.is_available()
                self.device_count = torch.cuda.device_count()
                self.device_name = (
                    torch.cuda.get_device_name(0) if self.cuda_available else None
                )

            def probe(self) -> ProbeResult:
                return {
                    "pid": self.pid,
                    "CUDA_VISIBLE_DEVICES": self.visible,
                    "cuda_available": self.cuda_available,
                    "device_count": self.device_count,
                    "device_name": self.device_name,
                }

        actors = [WarmTorchProbe.remote() for _ in range(2)]
        results = ray.get([actor.probe.remote() for actor in actors], timeout=240)
        pprint.pp(results)
        assert len(results) == 2, results
        assert all(result["cuda_available"] for result in results), results
        visible = [result["CUDA_VISIBLE_DEVICES"] for result in results]
        assert len(set(visible)) == 2, visible
        print("RAY_GPU_WARMUP_PROBE_OK")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
