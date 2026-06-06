#!/usr/bin/env python3
# pyright: reportAny=false, reportDeprecated=false, reportExplicitAny=false, reportUnannotatedClassAttribute=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnusedCallResult=false
"""Reusable OpenAI-compatible VLM judge client.

The deployed judge endpoints sit behind Cloudflare and reject default Python
HTTP user-agents.  This client therefore always sends a browser-like UA and can
use curl as the transport (default ``auto`` tries urllib first, then curl).

All public scoring methods are fail-soft: bad files, HTTP failures, malformed
JSON, and unparsable model answers return ``None`` instead of raising.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence, cast


Choice = Literal["A", "B"]
Transport = Literal["auto", "urllib", "curl"]

DEFAULT_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

QUALITY_PROMPT = (
    "Rate perceptual SR quality 1-10 (sharp/natural vs blurry/artifacted). "
    "Answer ONLY the integer."
)

GROUNDING_PROMPT = (
    "Here is an image and a text description. Does the text FAITHFULLY and "
    "ACCURATELY describe what is actually visible (no hallucinated/unrelated "
    "concepts)? Rate 1-10 (10=fully faithful; 1=describes unrelated things). "
    "Answer ONLY the integer.\n\nText description: {prompt}"
)

PAIRWISE_PROMPT = (
    "Two super-resolution crops are shown, labeled Image A and Image B. "
    "Which crop has higher perceptual quality (sharp, natural, fewer artifacts)? "
    "Answer ONLY A or B."
)

PROMPT_PAIRWISE_GROUNDING_PROMPT = (
    "Here is one image and two candidate text descriptions, labeled A and B. "
    "Which text description more faithfully and accurately describes what is "
    "actually visible in the image, with fewer hallucinated or unrelated concepts? "
    "Answer ONLY A or B."
)


@dataclass(frozen=True)
class JudgeResponse:
    text: str
    raw: dict[str, Any]


class VlmJudge:
    """Small fail-soft VLM judge for quality, grounding, and pairwise quality."""

    _NUMBER_RE: re.Pattern[str] = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")
    _CHOICE_RE: re.Pattern[str] = re.compile(r"\b([AB])\b", re.IGNORECASE)

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        ua: str | None = None,
        timeout: float | None = None,
        retries: int | None = None,
        transport: Transport | str | None = None,
        seed: int | None = 0,
        verbose: bool = False,
    ) -> None:
        self.base_url = (base_url or os.getenv("VLM_JUDGE_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.getenv("VLM_JUDGE_API_KEY") or ""
        self.model = model or os.getenv("VLM_JUDGE_MODEL") or ""
        self.ua = ua or os.getenv("VLM_JUDGE_UA") or DEFAULT_BROWSER_UA
        self.timeout = float(timeout if timeout is not None else os.getenv("VLM_JUDGE_TIMEOUT", "45"))
        self.retries = max(0, int(retries if retries is not None else os.getenv("VLM_JUDGE_RETRIES", "1")))
        selected_transport = transport or os.getenv("VLM_JUDGE_TRANSPORT", "auto")
        if selected_transport not in {"auto", "urllib", "curl"}:
            selected_transport = "auto"
        self.transport = cast(Transport, selected_transport)
        self.verbose = verbose
        self._rng = random.Random(seed)

    @classmethod
    def from_env(cls, prefix: str = "VLM_JUDGE", **overrides: Any) -> "VlmJudge":
        """Construct from ``<prefix>_*`` env vars, with explicit overrides."""

        def env(name: str) -> str | None:
            return os.getenv(f"{prefix}_{name}")

        kwargs: dict[str, Any] = {
            "base_url": env("BASE_URL"),
            "api_key": env("API_KEY"),
            "model": env("MODEL"),
            "ua": env("UA"),
            "timeout": env("TIMEOUT"),
            "retries": env("RETRIES"),
            "transport": env("TRANSPORT"),
        }
        kwargs.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**kwargs)

    def quality(self, image_path: str | Path) -> float | None:
        """Rate perceptual SR quality from 1 to 10."""

        content = self._image_prompt_content(QUALITY_PROMPT, image_path)
        if content is None:
            return None
        response = self._complete(content, max_tokens=16)
        if response is None:
            return None
        return self._parse_rating(response.text)

    def grounding(self, prompt_text: str, image_path: str | Path) -> float | None:
        """Rate prompt<->image faithfulness from 1 to 10."""

        prompt = GROUNDING_PROMPT.format(prompt=prompt_text.strip())
        content = self._image_prompt_content(prompt, image_path)
        if content is None:
            return None
        response = self._complete(content, max_tokens=16)
        if response is None:
            return None
        return self._parse_rating(response.text)

    def pairwise(self, imgA_path: str | Path, imgB_path: str | Path) -> Choice | None:
        """Return caller-mapped winner (``"A"`` or ``"B"``) for image quality.

        The display order is randomized internally to reduce position bias; the
        returned label always maps back to the caller's original A/B arguments.
        """

        display: list[tuple[Choice, Choice, str | Path]] = [
            ("A", "A", imgA_path),
            ("B", "B", imgB_path),
        ]
        if self._rng.random() < 0.5:
            display = [("A", "B", imgB_path), ("B", "A", imgA_path)]

        content: list[dict[str, Any]] = [{"type": "text", "text": PAIRWISE_PROMPT}]
        for display_label, _caller_label, path in display:
            data_url = self._image_data_url(path)
            if data_url is None:
                return None
            content.extend(
                [
                    {"type": "text", "text": f"Image {display_label}:"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]
            )
        content.append({"type": "text", "text": "Higher-quality crop: "})

        response = self._complete(content, max_tokens=8)
        if response is None:
            return None
        display_winner = self._parse_choice(response.text)
        if display_winner is None:
            return None
        for display_label, caller_label, _path in display:
            if display_label == display_winner:
                return caller_label
        return None

    def prompt_pairwise_grounding(
        self,
        promptA_text: str,
        promptB_text: str,
        image_path: str | Path,
    ) -> Choice | None:
        """Return which caller prompt (A/B) better describes one image.

        This is not used by the primary scalar grounding metric, but is useful as
        a quick alternative if absolute grounding scores saturate.  Prompt order
        is randomized internally and the returned label maps back to caller A/B.
        """

        data_url = self._image_data_url(image_path)
        if data_url is None:
            return None

        display: list[tuple[Choice, Choice, str]] = [
            ("A", "A", promptA_text),
            ("B", "B", promptB_text),
        ]
        if self._rng.random() < 0.5:
            display = [("A", "B", promptB_text), ("B", "A", promptA_text)]

        prompt_lines = [PROMPT_PAIRWISE_GROUNDING_PROMPT]
        for display_label, _caller_label, text in display:
            prompt_lines.append(f"Description {display_label}: {text.strip()}")
        prompt_lines.append("Better description: ")
        content = [
            {"type": "text", "text": "\n\n".join(prompt_lines)},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        response = self._complete(content, max_tokens=8)
        if response is None:
            return None
        display_winner = self._parse_choice(response.text)
        if display_winner is None:
            return None
        for display_label, caller_label, _text in display:
            if display_label == display_winner:
                return caller_label
        return None

    def _image_prompt_content(self, prompt: str, image_path: str | Path) -> list[dict[str, Any]] | None:
        data_url = self._image_data_url(image_path)
        if data_url is None:
            return None
        return [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]

    def _image_data_url(self, image_path: str | Path) -> str | None:
        try:
            path = Path(image_path)
            raw = path.read_bytes()
        except OSError as exc:
            self._log(f"failed to read image {image_path}: {exc}")
            return None
        mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _complete(self, content: list[dict[str, Any]], max_tokens: int) -> JudgeResponse | None:
        if not self.base_url or not self.api_key or not self.model:
            self._log("missing judge configuration: base_url, api_key, and model are required")
            return None
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        for attempt in range(self.retries + 1):
            raw = self._post_json(payload)
            if raw is not None:
                text = self._extract_text(raw)
                if text is not None:
                    return JudgeResponse(text=text.strip(), raw=raw)
                self._log(f"could not extract text from response: {raw}")
            if attempt < self.retries:
                time.sleep(min(4.0, 0.75 * (attempt + 1)))
        return None

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if self.transport in {"auto", "urllib"}:
            raw = self._post_json_urllib(payload)
            if raw is not None or self.transport == "urllib":
                return raw
            self._log("urllib failed; falling back to curl")
        return self._post_json_curl(payload)

    def _post_json_urllib(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        endpoint = f"{self.base_url}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - configured endpoint
                text = response.read().decode("utf-8", errors="replace")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            detail = ""
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:500]
                except OSError:
                    detail = ""
            self._log(f"urllib request failed: {exc} {detail}".strip())
            return None
        return self._loads_json(text)

    def _post_json_curl(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        endpoint = f"{self.base_url}/chat/completions"
        body = json.dumps(payload)
        args: list[str] = [
            "curl",
            "--silent",
            "--show-error",
            "--max-time",
            str(max(1, int(math.ceil(self.timeout)))),
            "--write-out",
            "\n__HTTP_STATUS__:%{http_code}",
            "--request",
            "POST",
            endpoint,
            "--header",
            "Content-Type: application/json",
            "--header",
            "Accept: application/json",
            "--header",
            f"Authorization: Bearer {self.api_key}",
            "--header",
            f"User-Agent: {self.ua}",
            "--data-binary",
            "@-",
        ]
        try:
            completed = subprocess.run(  # noqa: S603 - curl args are fixed, no shell
                args,
                input=body,
                text=True,
                capture_output=True,
                timeout=self.timeout + 5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._log(f"curl request failed: {exc}")
            return None
        if completed.returncode != 0:
            stderr = completed.stderr.strip()[:500]
            stdout = completed.stdout.strip()[:500]
            self._log(f"curl exited {completed.returncode}: {stderr} {stdout}".strip())
            return None
        body, status = self._split_curl_status(completed.stdout)
        if status is not None and not (200 <= status < 300):
            self._log(f"curl HTTP {status}: {body.strip()[:500]}")
            return None
        return self._loads_json(body)

    @staticmethod
    def _split_curl_status(stdout: str) -> tuple[str, int | None]:
        marker = "\n__HTTP_STATUS__:"
        if marker not in stdout:
            return stdout, None
        body, raw_status = stdout.rsplit(marker, 1)
        try:
            status = int(raw_status.strip())
        except ValueError:
            status = None
        return body, status

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self.ua,
        }

    def _loads_json(self, text: str) -> dict[str, Any] | None:
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            self._log(f"JSON parse failed: {exc}; body={text[:500]!r}")
            return None
        if not isinstance(raw, dict):
            self._log(f"unexpected JSON type: {type(raw).__name__}")
            return None
        return raw

    @classmethod
    def _extract_text(cls, raw: dict[str, Any]) -> str | None:
        try:
            choice0 = raw["choices"][0]
        except (KeyError, IndexError, TypeError):
            return None
        if isinstance(choice0, dict):
            message = choice0.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                text = cls._content_to_text(content)
                if text is not None:
                    return text
            text_field = choice0.get("text")
            if isinstance(text_field, str):
                return text_field
        return None

    @staticmethod
    def _content_to_text(content: object) -> str | None:
        if isinstance(content, str):
            return content
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    value = item.get("text")
                    if isinstance(value, str):
                        parts.append(value)
            if parts:
                return "\n".join(parts)
        return None

    @classmethod
    def _parse_rating(cls, text: str) -> float | None:
        match = cls._NUMBER_RE.search(text.strip())
        if match is None:
            return None
        try:
            value = float(match.group(0))
        except (OverflowError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        return max(1.0, min(10.0, value))

    @classmethod
    def _parse_choice(cls, text: str) -> Choice | None:
        stripped = text.strip().upper()
        if stripped in {"A", "B"}:
            return cast(Choice, stripped)
        match = cls._CHOICE_RE.search(stripped)
        if match is None:
            return None
        return cast(Choice, match.group(1).upper())

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[VlmJudge] {message}", file=sys.stderr)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one VLM judge query.")
    parser.add_argument("--base-url", default=os.getenv("VLM_JUDGE_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("VLM_JUDGE_API_KEY"))
    parser.add_argument("--model", default=os.getenv("VLM_JUDGE_MODEL"))
    parser.add_argument("--ua", default=os.getenv("VLM_JUDGE_UA", DEFAULT_BROWSER_UA))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("VLM_JUDGE_TIMEOUT", "45")))
    parser.add_argument("--retries", type=int, default=int(os.getenv("VLM_JUDGE_RETRIES", "1")))
    parser.add_argument("--transport", choices=("auto", "urllib", "curl"), default=os.getenv("VLM_JUDGE_TRANSPORT", "auto"))
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    quality = sub.add_parser("quality", help="score one image's perceptual quality")
    quality.add_argument("image", type=Path)

    grounding = sub.add_parser("grounding", help="score prompt/image faithfulness")
    grounding.add_argument("prompt")
    grounding.add_argument("image", type=Path)

    pairwise = sub.add_parser("pairwise", help="choose higher-quality image")
    pairwise.add_argument("image_a", type=Path)
    pairwise.add_argument("image_b", type=Path)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    judge = VlmJudge(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        ua=args.ua,
        timeout=args.timeout,
        retries=args.retries,
        transport=args.transport,
        verbose=args.verbose,
    )
    if args.cmd == "quality":
        print(judge.quality(args.image))
    elif args.cmd == "grounding":
        print(judge.grounding(args.prompt, args.image))
    elif args.cmd == "pairwise":
        print(judge.pairwise(args.image_a, args.image_b))


if __name__ == "__main__":
    main()
