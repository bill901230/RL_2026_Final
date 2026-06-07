from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse, unquote

from PIL import Image


def _open_image(value):
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if hasattr(value, "read"):
        return Image.open(value).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(BytesIO(value)).convert("RGB")
    if isinstance(value, str):
        parsed = urlparse(value)
        if parsed.scheme == "file":
            return Image.open(Path(unquote(parsed.path))).convert("RGB")
        return Image.open(Path(value)).convert("RGB")
    raise TypeError(f"Unsupported image input type: {type(value)!r}")


def fetch_image(ele, size_factor=28):
    if isinstance(ele, dict):
        if "image" in ele:
            return _open_image(ele["image"])
        if "path" in ele:
            return _open_image(ele["path"])
        if "bytes" in ele:
            return _open_image(ele["bytes"])
    return _open_image(ele)


def fetch_video(ele, *args, **kwargs):
    raise NotImplementedError("W3.T1 smoke shim only supports images, not video.")
