# pyright: reportMissingImports=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

import sys
from pathlib import Path


EVIDENCE_DIR = Path(__file__).resolve().parent
sys.path = [
    path
    for path in sys.path
    if path and Path(path).resolve() != EVIDENCE_DIR
]

from vllm import LLM, SamplingParams


def main() -> None:
    llm = LLM(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.70,
        enforce_eager=True,
        trust_remote_code=True,
        max_model_len=2048,
    )
    outputs = llm.generate(
        ["Describe a small red cube in one short sentence."],
        SamplingParams(max_tokens=16, temperature=0.0),
    )
    text = outputs[0].outputs[0].text.strip()
    print(text)
    assert text, outputs
    print("VLLM_ROLLOUT_OK")


if __name__ == "__main__":
    main()
