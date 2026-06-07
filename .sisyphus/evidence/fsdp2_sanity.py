# pyright: reportMissingImports=false, reportUnknownMemberType=false

import os

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    model = torch.nn.Linear(4, 4, bias=False).cuda()
    fully_shard(model)
    out = model(torch.randn(2, 4, device="cuda")).sum()
    out.backward()
    torch.cuda.synchronize()
    dist.destroy_process_group()
    print("FSDP2_SANITY_OK")


if __name__ == "__main__":
    main()
