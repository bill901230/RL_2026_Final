import importlib
from pathlib import Path
from typing import Any

import pytest
import torch
from _pytest.monkeypatch import MonkeyPatch

text_sim = importlib.import_module("train.grpo.text_sim")
ClipEmbedder = text_sim.ClipEmbedder
ngram_overlap = text_sim.ngram_overlap


class _FakeBatch(dict[str, Any]):
    def to(self, device: str) -> "_FakeBatch":
        self["device"] = device
        return self


class _FakeProcessor:
    def __call__(
        self,
        text: list[str] | None = None,
        return_tensors: str | None = None,
        padding: bool | None = None,
        truncation: bool | None = None,
    ) -> _FakeBatch:
        assert return_tensors == "pt"
        assert padding is True
        assert truncation is True
        return _FakeBatch({"text": text})


class _FakeModel:
    device = ""

    def to(self, device: str) -> "_FakeModel":
        self.device = device
        return self

    def eval(self) -> "_FakeModel":
        return self

    def parameters(self) -> list[Any]:
        return []

    def get_text_features(self, text: list[str], device: str | None = None) -> torch.Tensor:
        _ = device
        vectors = {
            "eye anchor texture": torch.tensor([1.0, 0.0, 0.0]),
            "unrelated neural synapse": torch.tensor([-1.0, 0.0, 0.0]),
        }
        return torch.stack([vectors[t] for t in text])


def test_cosine_maps_raw_clip_cosine_to_unit_interval():
    anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    reference = torch.tensor([[1.0, 0.0]])

    similarities = ClipEmbedder.cosine(anchors, reference).flatten().tolist()

    assert similarities == pytest.approx([1.0, 0.5, 0.0])


def test_ngram_overlap_jaccard_contracts():
    text = "red blue green"
    other = "red blue yellow"

    identical = ngram_overlap(text, [text], n=2)
    disjoint = ngram_overlap(text, ["alpha beta gamma"], n=2)
    partial = ngram_overlap(text, [other], n=2)

    assert identical == pytest.approx(1.0)
    assert disjoint == pytest.approx(0.0)
    assert 0.0 <= partial <= 1.0
    assert partial == pytest.approx(ngram_overlap(other, [text], n=2))


def test_anchor_contract_with_stubbed_clip_embeddings(monkeypatch: MonkeyPatch):
    model_ids: list[tuple[str, str]] = []

    class _ModelFactory:
        @staticmethod
        def from_pretrained(model_id: str) -> _FakeModel:
            model_ids.append(("model", model_id))
            return _FakeModel()

    class _ProcessorFactory:
        @staticmethod
        def from_pretrained(model_id: str) -> _FakeProcessor:
            model_ids.append(("processor", model_id))
            return _FakeProcessor()

    monkeypatch.setattr(text_sim, "CLIPModel", _ModelFactory)
    monkeypatch.setattr(text_sim, "CLIPProcessor", _ProcessorFactory)

    clip = ClipEmbedder(device="cpu")
    anchor = clip.embed_text("eye anchor texture")
    same = clip.embed_text("eye anchor texture")
    unrelated = clip.embed_text("unrelated neural synapse")

    same_sim = clip.cosine(anchor, same).item()
    unrelated_sim = clip.cosine(anchor, unrelated).item()

    assert model_ids == [
        ("model", "openai/clip-vit-base-patch32"),
        ("processor", "openai/clip-vit-base-patch32"),
    ]
    assert same_sim == pytest.approx(1.0)
    assert same_sim > unrelated_sim
    assert unrelated_sim == pytest.approx(0.0)


def test_repetition_overlap_penalizes_identical_more_than_disjoint():
    prompt = "neurons dendrites axons synapses"

    repeated = ngram_overlap(prompt, [prompt], n=2)
    disjoint = ngram_overlap(prompt, ["brick window roof tile"], n=2)

    assert repeated > disjoint
    assert repeated == pytest.approx(1.0)
    assert disjoint == pytest.approx(0.0)


def test_0064_repetition():
    root = Path(__file__).resolve().parents[1]
    prompt_2 = root / "fail_case/failcase_1/coz_output/per-sample/0064_eye/txt/2.txt"
    prompt_3 = root / "fail_case/failcase_1/coz_output/per-sample/0064_eye/txt/3.txt"

    overlap = ngram_overlap(prompt_2.read_text().strip(), [prompt_3.read_text().strip()], n=2)

    assert overlap > 0.5
