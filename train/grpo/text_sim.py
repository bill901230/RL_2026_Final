"""Shared CLIP embedders for the text/image similarity rewards.

Used by:
  - R_anc : cosine(prompt, x_0 caption)            -> text-text
  - R_rep : cosine(prompt, previous-scale prompts) -> text-text
  - R_fb  : cosine(SR image, input crop)           -> image-image (consistency)
  - R_crit (clipscore backend): cosine(image, prompt) -> image-text

One CLIP model serves all of them. ngram_overlap is a cheap lexical companion
to the embedding similarity for R_rep.
"""
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


class ClipEmbedder:
    def __init__(self, model_id="openai/clip-vit-base-patch32", device="cuda"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def embed_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        feats = self.model.get_text_features(**inputs)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def embed_image(self, images):
        """images: list of PIL.Image."""
        if not isinstance(images, (list, tuple)):
            images = [images]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        return F.normalize(feats, dim=-1)

    @staticmethod
    def cosine(a, b):
        """a:(N,D) b:(M,D) normalised -> (N,M) cosine similarity."""
        return a @ b.t()


def ngram_set(text, n=2):
    tokens = [t for t in text.lower().replace(",", " ").split() if t]
    if len(tokens) < n:
        return set(tokens)
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def ngram_overlap(text, others, n=2):
    """Max Jaccard n-gram overlap between `text` and any string in `others`."""
    a = ngram_set(text, n)
    if not a or not others:
        return 0.0
    best = 0.0
    for o in others:
        b = ngram_set(o, n)
        if not b:
            continue
        inter = len(a & b)
        union = len(a | b)
        if union:
            best = max(best, inter / union)
    return best
