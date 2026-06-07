# pyright: reportMissingImports=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

import os
import sys
from pathlib import Path


EVIDENCE_DIR = Path(__file__).resolve().parent
sys.path = [
    path
    for path in sys.path
    if path and Path(path).resolve() != EVIDENCE_DIR
]

import ray
from hydra import compose, initialize_config_dir

import verl
from verl.trainer.main_ppo import run_ppo


def main() -> None:
    config_dir = Path(verl.__file__).resolve().parent / "trainer" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=sys.argv[1:])

    if not ray.is_initialized():
        ray.init(
            _temp_dir=os.environ["RAY_TMPDIR"],
            include_dashboard=False,
            object_store_memory=8 * 1024**3,
            num_cpus=config.ray_init.num_cpus,
        )

    run_ppo(config)


if __name__ == "__main__":
    main()
