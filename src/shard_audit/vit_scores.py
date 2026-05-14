"""Per-image membership scoring for Vision Transformer audits.

Membership signal definition (analogue of MIN-K log-prob in the LLM pipeline):

    score(x, y) = log P_head( T_encoder(x) )[y]

where
- T_encoder is the encoder of the candidate target model (the model being audited),
- P_head is the classifier head of the parent model (always ImageNet-1k 1000-way),
- y is the ground-truth ImageNet class for image x.

Higher score = the target model assigns higher probability to the correct class
when scored by the parent's head. For a model that inherits the parent's encoder
(LoRA / continued finetune), member images should retain higher scores than
nonmembers; for an independent encoder (e.g., random init) the parent's head
sees an unrelated feature distribution and the score is near-uniform.

For the parent itself we have T_encoder == P_encoder, recovering ordinary
log p_P(y | x) — the pretraining loss on the actual training label.

We also compute several auxiliary scores for diagnostics:
- mean_loss        : -score (the standard CE loss); lower = more member-like
- top1_logprob_own : log p_T(top-class)  using T's own head; calibration-free
                     proxy that works even without an ImageNet-aligned head
- pred_entropy     : entropy of T's own softmax; lower = more confident = more member-like
                     (we negate so higher = more member-like, see code)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class ScoreBundle:
    """Per-sample scores returned by :func:`score_batch`. All keys follow the
    convention 'higher = more member-like' so they plug straight into the
    threshold distinguisher in run_mia_experiment.py."""
    p_on_t_logprob: float    # log P_head( T_encoder(x) )[y]
    p_on_t_loss: float       # -p_on_t_logprob
    t_own_top1_logprob: float
    t_own_neg_entropy: float


def encoder_features(target_model, pixel_values: torch.Tensor) -> torch.Tensor:
    """Return T's [CLS] feature ([B, hidden]) regardless of architecture.

    Works for HF ViTForImageClassification, DeiTForImageClassification, and
    similar that expose a `vit` / `deit` / `base_model` submodule.
    """
    for attr in ("vit", "deit", "base_model"):
        if hasattr(target_model, attr) and getattr(target_model, attr) is not None:
            mod = getattr(target_model, attr)
            try:
                out = mod(pixel_values=pixel_values)
            except TypeError:
                out = mod(pixel_values)
            if hasattr(out, "last_hidden_state"):
                return out.last_hidden_state[:, 0, :]
            if isinstance(out, (tuple, list)):
                return out[0][:, 0, :]
            return out[:, 0, :]
    # Fallback: call the model and pull CLS from any attribute we find.
    raise RuntimeError(
        f"Could not locate ViT/DeiT encoder on model of type {type(target_model).__name__}"
    )


def score_batch(
    images: list[Image.Image],
    labels: torch.Tensor,                     # [B], long, ImageNet 0..999
    target_model,                             # T (the model we are auditing)
    parent_head: torch.nn.Linear,             # P.classifier — always 768->1000 here
    image_processor,                          # HF AutoImageProcessor for the *parent*
    target_processor=None,                    # optional: T's own preprocessor for "own" scores
    target_uses_imagenet_head: bool = False,  # if True, use T's own head for "own" scores
    device: str = "cuda",
) -> list[ScoreBundle]:
    """Compute the four per-sample scores for a batch of PIL images."""
    target_model.eval()

    # 1. Tokenize for the parent / cross-head score (uses parent's processor so
    # the resize/normalize matches what P's classifier expects).
    parent_inputs = image_processor(images=images, return_tensors="pt").to(device)
    labels = labels.to(device)

    with torch.no_grad():
        feats = encoder_features(target_model, parent_inputs["pixel_values"])  # [B, 768]
        # Cross-head score: P's head on T's encoder
        cross_logits = parent_head(feats)                                       # [B, 1000]
        cross_logp = F.log_softmax(cross_logits, dim=-1)
        gathered = cross_logp.gather(1, labels.unsqueeze(1)).squeeze(1)         # [B]
        p_on_t_logprob = gathered.float().cpu()

        # 2. T's own predictions, using T's own preprocessor if available.
        if target_processor is not None:
            own_inputs = target_processor(images=images, return_tensors="pt").to(device)
        else:
            own_inputs = parent_inputs
        own_out = target_model(**own_inputs)
        own_logp_full = F.log_softmax(own_out.logits, dim=-1)                   # [B, K_T]

        # Top-1 log-prob under T's own head (K_T may not equal 1000)
        own_top1_lp, _ = own_logp_full.max(dim=-1)
        own_top1_lp = own_top1_lp.float().cpu()

        # Negative entropy of T's softmax (higher = more confident = member-like)
        own_p = own_logp_full.exp()
        own_entropy = -(own_p * own_logp_full).sum(dim=-1)                      # [B]
        own_neg_entropy = (-own_entropy).float().cpu()

    out: list[ScoreBundle] = []
    for i in range(len(images)):
        out.append(ScoreBundle(
            p_on_t_logprob=float(p_on_t_logprob[i].item()),
            p_on_t_loss=float(-p_on_t_logprob[i].item()),
            t_own_top1_logprob=float(own_top1_lp[i].item()),
            t_own_neg_entropy=float(own_neg_entropy[i].item()),
        ))
    return out


def load_parent_head(
    parent_model_id: str,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
) -> tuple[torch.nn.Linear, "AutoImageProcessor"]:
    """Load the parent's classifier head (Linear) and its image processor."""
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    P = AutoModelForImageClassification.from_pretrained(
        parent_model_id,
        torch_dtype=dtype,
    ).to(device)
    P.eval()
    if not isinstance(P.classifier, torch.nn.Linear):
        raise RuntimeError(
            f"Parent {parent_model_id} has classifier of type {type(P.classifier).__name__}; "
            "expected Linear."
        )
    proc = AutoImageProcessor.from_pretrained(parent_model_id, use_fast=True)
    return P.classifier, proc


def make_random_init_model(
    reference_model_id: str,
    seed: int = 0,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
):
    """Build a fresh, randomly-initialised ViT with the same architecture as
    ``reference_model_id``. Useful as a negative control: same shape as P, but
    no path from any training data to its parameters."""
    from transformers import AutoConfig, AutoModelForImageClassification
    cfg = AutoConfig.from_pretrained(reference_model_id)
    # transformers uses torch's global RNG for init; set seed for reproducibility.
    torch.manual_seed(seed)
    model = AutoModelForImageClassification.from_config(cfg)
    if dtype is not None:
        model = model.to(dtype)
    return model.to(device).eval()
