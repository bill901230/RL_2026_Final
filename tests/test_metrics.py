import importlib
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Protocol, cast

import pytest
import torch

import pyiqa  # pyright: ignore[reportMissingTypeStubs]


class _MetricScores(Protocol):
    def score(self, name: str, img: torch.Tensor) -> torch.Tensor: ...

    def score_all(self, img: torch.Tensor) -> dict[str, torch.Tensor]: ...


class _IQAMetricsClass(Protocol):
    def __call__(self, metric_names: Iterable[str], device: str = "cuda") -> _MetricScores: ...


class _MetricsModule(Protocol):
    pyiqa: object
    HIGHER_BETTER: dict[str, bool]
    IQAMetrics: _IQAMetricsClass


metrics_mod = cast(_MetricsModule, cast(object, importlib.import_module("train.grpo.metrics")))
HIGHER_BETTER = metrics_mod.HIGHER_BETTER
IQAMetrics = metrics_mod.IQAMetrics


METRIC_NAMES = ("niqe", "musiq", "maniqa", "clipiqa")
FAKE_RAW_SCORES = {
    "niqe": 12.0,
    "musiq": 70.0,
    "maniqa": 0.6,
    "clipiqa": 0.7,
}


class _FakeMetric:
    def __init__(self, name: str):
        self.name: str = name

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return torch.full((img.shape[0],), FAKE_RAW_SCORES[self.name], device=img.device)


class _LiveMetric(Protocol):
    lower_better: bool


class _CreateMetric(Protocol):
    def __call__(self, metric_name: str, *, device: str) -> _LiveMetric: ...


@pytest.fixture()
def fake_pyiqa(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    created: list[tuple[str, str]] = []

    def create_metric(name: str, device: str) -> _FakeMetric:
        created.append((name, device))
        return _FakeMetric(name)

    monkeypatch.setattr(metrics_mod, "pyiqa", SimpleNamespace(create_metric=create_metric))
    return created


def test_score_accepts_nchw_float_unit_interval(fake_pyiqa: list[tuple[str, str]]) -> None:
    metrics = IQAMetrics(["musiq"], device="cpu")
    img = torch.full((2, 3, 8, 8), 0.5)

    score = metrics.score("musiq", img)

    assert fake_pyiqa == [("musiq", "cpu")]
    torch.testing.assert_close(score, torch.full((2,), FAKE_RAW_SCORES["musiq"]))


@pytest.mark.parametrize(
    ("img", "match"),
    [
        pytest.param(torch.rand(8, 8, 3), "NCHW", id="hwc"),
        pytest.param(torch.ones((1, 3, 8, 8), dtype=torch.uint8), "floating", id="uint8"),
        pytest.param(torch.full((1, 3, 8, 8), 255.0), r"\[0, 1\]", id="255-range"),
    ],
)
def test_score_rejects_inputs_outside_documented_contract(
    fake_pyiqa: list[tuple[str, str]], img: torch.Tensor, match: str
) -> None:
    metrics = IQAMetrics(["musiq"], device="cpu")
    assert fake_pyiqa == [("musiq", "cpu")]

    with pytest.raises((TypeError, ValueError), match=match):
        _ = metrics.score("musiq", img)


def test_score_all_returns_all_metrics_oriented_higher_better(
    fake_pyiqa: list[tuple[str, str]],
) -> None:
    metrics = IQAMetrics(METRIC_NAMES, device="cpu")
    img = torch.full((2, 3, 8, 8), 0.25)

    scores = metrics.score_all(img)

    assert fake_pyiqa == [(name, "cpu") for name in METRIC_NAMES]
    assert set(scores) == set(METRIC_NAMES)
    torch.testing.assert_close(scores["niqe"], torch.full((2,), -FAKE_RAW_SCORES["niqe"]))
    for name in ("musiq", "maniqa", "clipiqa"):
        torch.testing.assert_close(scores[name], torch.full((2,), FAKE_RAW_SCORES[name]))


@pytest.mark.parametrize("name", METRIC_NAMES)
def test_map_matches_pyiqa(name: str) -> None:
    create_metric = cast(_CreateMetric, getattr(pyiqa, "create_metric"))
    live_metric = create_metric(name, device="cpu")
    live_higher_better = not bool(live_metric.lower_better)

    assert HIGHER_BETTER[name] == live_higher_better


def test_niqe_is_flagged_lower_better_for_raw_score_inversion() -> None:
    assert HIGHER_BETTER["niqe"] is False
    assert all(HIGHER_BETTER[name] is True for name in ("musiq", "maniqa", "clipiqa"))
