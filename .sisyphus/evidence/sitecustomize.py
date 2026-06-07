# pyright: reportMissingSuperCall=false, reportAttributeAccessIssue=false
"""W3.T1 smoke-run compatibility shims loaded via PYTHONPATH.

The isolated training venv pairs datasets==2.14.4 with pyarrow==24.0.0.
datasets 2.14 still references the removed pyarrow.PyExtensionType symbol, so
importing verl.utils.dataset.rl_dataset fails before training starts.  Keep the
shim local to this smoke by adding .sisyphus/evidence to PYTHONPATH for the run.
"""

try:
    import pyarrow as pa

    if not hasattr(pa, "PyExtensionType"):

        class _CompatPyExtensionType(pa.ExtensionType):
            def __init__(self, storage_type):
                pa.ExtensionType.__init__(self, storage_type, "arrow.py_extension_type")

            def __arrow_ext_serialize__(self):
                return b""

            @classmethod
            def __arrow_ext_deserialize__(cls, storage_type, serialized):
                return cls(storage_type)

            def __reduce__(self):
                return _CompatPyExtensionType, (self.storage_type,)

        pa.PyExtensionType = _CompatPyExtensionType
except Exception:
    pass

try:
    from fsspec.implementations.local import LocalFileSystem

    if isinstance(LocalFileSystem.protocol, (tuple, list)):
        LocalFileSystem.protocol = "file"
except Exception:
    pass

try:
    import os

    if os.environ.get("W3T1_FORCE_SDPA", "1"):
        from transformers import AutoConfig

        _orig_auto_config_from_pretrained = AutoConfig.from_pretrained

        @classmethod
        def _w3t1_auto_config_from_pretrained(cls, *args, **kwargs):
            if kwargs.get("attn_implementation") == "flash_attention_2":
                kwargs["attn_implementation"] = "sdpa"
                print("W3T1_FORCE_SDPA replaced flash_attention_2 with sdpa", flush=True)
            return _orig_auto_config_from_pretrained(*args, **kwargs)

        AutoConfig.from_pretrained = _w3t1_auto_config_from_pretrained

    if os.environ.get("W3T1_TRACE_DIST"):
        import datetime
        import torch.distributed as _dist

        _orig_init_process_group = _dist.init_process_group

        def _w3t1_init_process_group(*args, **kwargs):
            kwargs.setdefault("timeout", datetime.timedelta(seconds=90))
            print(
                "W3T1_DIST_INIT"
                f" pid={os.getpid()} rank={os.environ.get('RANK')} world={os.environ.get('WORLD_SIZE')}"
                f" local_rank={os.environ.get('LOCAL_RANK')} ray_local_rank={os.environ.get('RAY_LOCAL_RANK')}"
                f" master={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}"
                f" cuda={os.environ.get('CUDA_VISIBLE_DEVICES')} rocr={os.environ.get('ROCR_VISIBLE_DEVICES')}"
                f" backend={kwargs.get('backend') if 'backend' in kwargs else (args[0] if args else None)}",
                flush=True,
            )
            out = _orig_init_process_group(*args, **kwargs)
            print(f"W3T1_DIST_INIT_DONE pid={os.getpid()} rank={os.environ.get('RANK')}", flush=True)
            return out

        _dist.init_process_group = _w3t1_init_process_group
except Exception:
    pass
