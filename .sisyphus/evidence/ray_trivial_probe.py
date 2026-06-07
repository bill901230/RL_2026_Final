# pyright: reportAttributeAccessIssue=false, reportMissingImports=false
# pyright: reportMissingTypeArgument=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownParameterType=false
# pyright: reportUnknownVariableType=false, reportUntypedClassDecorator=false

import os
import pprint

import ray


def main() -> None:
    try:
        ray.init(
            _temp_dir=os.environ["RAY_TMPDIR"],
            include_dashboard=False,
            object_store_memory=8 * 1024**3,
            num_cpus=8,
        )

        @ray.remote(num_gpus=1, num_cpus=0)
        class TrivialProbe:
            def probe(self) -> dict[str, int | str]:
                import os

                return {
                    "pid": os.getpid(),
                    "CUDA_VISIBLE_DEVICES": os.environ.get(
                        "CUDA_VISIBLE_DEVICES", ""
                    ),
                }

        actors = [TrivialProbe.remote() for _ in range(2)]
        results = ray.get([actor.probe.remote() for actor in actors], timeout=60)
        pprint.pp(results)
        assert len(results) == 2, results
        visible = [result["CUDA_VISIBLE_DEVICES"] for result in results]
        assert len(set(visible)) == 2, visible
        print("RAY_TRIVIAL_PROBE_OK")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
