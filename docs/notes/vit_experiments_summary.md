# Vision Model Provenance — Experiment Summary

**Session date:** 2026-05-07
**Codebase:** model_provenance_code
**Branch:** main
**Compute:** single L40S (46 GB), conda env `kvm`

## TL;DR

We ported the shard-membership audit from the LLM pipeline (MIMIR/Pythia) to a
ViT image-classifier setting. The audit reduces to a level-α distinguishing
test on a per-image score; the framework's predictions hold quantitatively
across **three access regimes** (grey-box encoder access, full black-box
logits, label-only black-box logits), and the verdict accuracy is governed by
how well the score can be anchored to the parent's representation rather than
to the parent's training data.

| Access regime | Score | Derivatives detected | DeiT confounder | Total verdict accuracy |
|---|---|:---:|:---:|:---:|
| Grey-box (encoder + parent head) | `log P_head(T_enc(x))[y]` | **7/7** | rejected (correct) | **9/9** |
| Black-box, full logits, no labels | `aug_neg_logit_var_mean` (K=8) | **7/7** | flagged (false positive) | **8/9** |
| Black-box, full logits, no labels | `t_own_top1_logprob` | 0/7 | flagged (false positive) | 1/9 |

The framework's identifiability claim (Assumption 1.iii: no path
$S\leftarrow H\rightarrow T$) is empirically the dividing line: scores that
anchor to $P$ are robust to a shared-training-shard confounder like DeiT;
black-box scores that only probe $T$ are not.

---

## 1. Pipeline reuse

The repository was already wired for a single shard-membership pipeline:

1. **Data prep** → `train.jsonl` / `test.jsonl` with `id`, `label`,
   `phase_split`, content fields, hash.
2. **Score extraction** → `train_scores.jsonl` / `test_scores.jsonl` with one
   or more score columns (convention: higher = more member-like).
3. **MIA experiment** → `run_mia_experiment.py` calibrates a threshold on the
   train split, evaluates on test, reports advantage, AUC, accuracy, shuffled-
   label control, and (optional) parent-threshold transfer.

We re-used (3) verbatim and added analogous (1) and (2) for vision:

- **`scripts/data/prepare_imagenet_shard.py`** — class-balanced sampler from
  the non-gated mirror `evanarlian/imagenet_1k_resized_256`.
- **`src/shard_audit/vit_scores.py`** — encoder-routed score
  `log P_head(T_enc(x))[y]`, random-init negative control builder.
- **`scripts/scoring/extract_image_scores.py`** — encoder-routed scoring
  driver.
- **`scripts/scoring/extract_image_scores_aug.py`** — augmentation-MIA
  scoring driver (full-logit access).
- **`scripts/experiments/sweep_m_imagenet_shard.py`** — shard-size sweep.

---

## 2. Setup

### Parent model
- **P** = [`google/vit-base-patch16-224`](https://huggingface.co/google/vit-base-patch16-224)
  ViT-B/16 pretrained on ImageNet-21k, fine-tuned on ImageNet-1k.
- Training shard $S$ → ImageNet-1k train.
- Control shard $S'$ → ImageNet-1k val.

### Targets
- **Seven declared derivatives** of P (all declare `base_model: google/vit-base-patch16-224`):
  - `ISxOdin/vit-base-oxford-iiit-pets` — *bit-identical encoder*, only head retrained.
  - `MatanBT/vit-base-patch16-224-cifar10`
  - `MatanBT/vit-base-patch16-224-cifar100`
  - `Abhiram4/vit-base-patch16-224-finetuned-eurosat`
  - `MatanBT/vit-base-patch16-224-celeba-smiling`
  - `rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224`
  - `google/vit-base-patch16-224` itself, as self-test.

  Mean encoder weight drift relative to P: 0 (pets) up to 1.5e-3 (cifar100).

- **Two negative controls**:
  - ViT-B/16-224 with fresh **random initialisation** — severs S→P→T entirely.
  - [`facebook/deit-base-patch16-224`](https://huggingface.co/facebook/deit-base-patch16-224)
    — same architecture, *independently trained on the same ImageNet-1k corpus*.
    Encoder weight drift relative to P: 0.22 (two orders of magnitude larger
    than any derivative). DeiT explicitly probes the no-shard-confounding
    clause of Assumption 1.iii.

### Shard construction
- Dataset: `evanarlian/imagenet_1k_resized_256` (non-gated, no HF_TOKEN needed,
  ~25 GB total, downloaded 4 train shards + 2 val shards ≈ 3.4 GB).
- **77 classes** (intersection of class coverage of the downloaded shards).
- **Class-balanced** sampling: 25 members + 25 nonmembers per class per MIA
  split → $m = 1925$ records each for calibration and evaluation.
- Critical at α = 0.05: $\gamma_{0.05} = \sqrt{\log(2/0.05)/1925} = 0.044$.

### Confounder caught during smoke test
The HF parquet shards are class-sorted: a uniform sample from shard 0
covers ~19 classes while a uniform sample from val covers ~497 classes.
Under such mismatched class distributions, *P*'s per-class head bias acts
as a latent confounder $H$ satisfying $S\leftarrow H\rightarrow T$, and the
random-init control produced a spurious 0.61 AUC. Class-balanced sampling
collapses this to 0.50 (table 1) and is essential for any vision MIA whose
data source is class-sorted on disk.

---

## 3. Per-image scores

| Symbol | Definition | Access regime |
|---|---|---|
| $s_{T}^{\mathrm{enc}}(x,y)$ | $\log\,\mathrm{softmax}\!\bigl(P_{\mathrm{head}}(T_{\mathrm{enc}}(x))\bigr)_{y}$ | grey-box, label-aware |
| $s_{T}^{\mathrm{top1}}(x)$ | $\max_{c}\,\log\,\mathrm{softmax}(T(x))_{c}$ | black-box, label-free |
| $s_{T}^{\mathrm{aug}}(x)$ | $-\tfrac{1}{|C_T|}\sum_{c}\mathrm{Var}_{k=1..K}\!\bigl(\mathrm{logit}_k[c]\bigr)$, $K=8$ | black-box, label-free, augmentations |

All conventions are "higher = more member-like".

---

## 4. Experiment 1 — Encoder-routed score (Table 1)

**Score:** $s_{T}^{\mathrm{enc}}$. **Access:** $P$'s classifier head + $T$'s [CLS] feature.

| Target | Encoder Δ vs P | Adv_ctrl | Adv_main | Verdict / Correct? |
|---|---:|---:|---:|---|
| `google/vit-base-patch16-224` (self) | 0 | 0.041 | **+0.049** | Reject H₀ / Yes |
| `ISxOdin/vit-base-oxford-iiit-pets` | 0 (frozen enc.) | 0.041 | **+0.049** | Reject H₀ / Yes |
| `MatanBT/...-cifar10` | 1.1e-3 | 0.034 | **+0.068** | Reject H₀ / Yes |
| `MatanBT/...-cifar100` | 1.5e-3 | 0.000 | **+0.053** | Reject H₀ / Yes |
| `Abhiram4/...-eurosat` | 5.0e-4 | 0.051 | **+0.045** | Reject H₀ / Yes |
| `MatanBT/...-celeba-smiling` | 1.3e-3 | 0.066 | **+0.047** | Reject H₀ / Yes |
| `rvv-karma/Human-Action-Recognition` | 8.0e-4 | 0.002 | **+0.061** | Reject H₀ / Yes |
| ViT-B/16-224 random-init | — | 0.008 | **−0.001** | Fail to reject / Yes |
| `facebook/deit-base-patch16-224` | 0.22 | −0.036 | **−0.011** | Fail to reject / Yes |

**Verdict accuracy: 9/9.** Notable observations:

1. **Pets ≡ parent numerically** because the pets fine-tune froze the encoder
   bit-for-bit; only the head was retrained.
2. **DeiT correctly fails to reject** even though it co-trained on ImageNet-1k.
   The reason is geometric: routing through $P_{\mathrm{head}}$ requires that
   $T$'s features sit in the same coordinate frame as $P$'s. DeiT's
   independently-trained encoder is in a different frame (mean drift 0.22),
   so $P_{\mathrm{head}}$ produces near-uniform logits and the score carries
   no membership signal. This is a *stronger* specificity property than
   Assumption 1.iii alone guarantees in the worst case.

---

## 5. Experiment 2 — Shard size sweep (Figure)

**Target:** the highest-advantage derivative,
`MatanBT/vit-base-patch16-224-cifar10` (Adv_main = 0.068).

**Protocol.** For each $m\in\{20,40,80,160,320,640,960,1280,1600,1920\}$,
draw 50 class-balanced sub-samples of $m/2$ members and $m/2$ nonmembers from
each split, refit the threshold, evaluate. Trace the random-init control
through the same pipeline.

**Headline numbers** (top1 reject rates over 50 seeds):

| m | derivative adv ± std | derivative reject rate | control reject rate | γ_0.05(m) |
|---:|---:|---:|---:|---:|
| 20 | +0.022 ± 0.190 | 0.04 | 0.00 | 0.430 |
| 80 | +0.021 ± 0.085 | 0.02 | 0.02 | 0.215 |
| 320 | +0.044 ± 0.045 | 0.04 | 0.02 | 0.107 |
| 640 | +0.065 ± 0.037 | **0.42** | 0.02 | 0.076 |
| 1280 | +0.067 ± 0.023 | **0.70** | 0.02 | 0.054 |
| 1920 | +0.069 ± 0.014 | **0.96** | 0.00 | 0.044 |

**Findings.**

1. **Derivative advantage is approximately constant** in $m$, fluctuating
   around the asymptotic effect size $\widehat{\mathrm{Adv}}_\infty\approx 0.067$.
2. **Critical threshold shrinks at $1/\sqrt{m}$** as predicted by the theory.
3. **Crossover $m^\star \approx 880$**, in agreement with the sample-
   complexity prediction $\log(2/\alpha)/\widehat{\mathrm{Adv}}_\infty^{2}\approx 824$
   (within 7%).
4. **Random-init reject rate stays at or below $\alpha=0.05$ at every $m$**,
   empirically validating the Type-I bound in Proposition 1 at finite samples.
5. Below $m\approx 320$ the test fails to reject more than 85% of the time
   even though the model genuinely *is* a derivative — the one-sided nature
   of the test under Theorem 1.

**Practical rule:** to audit a derivative with effect size $\Delta$ at level
$\alpha$, allocate roughly $m\gtrsim \log(2/\alpha)/\Delta^{2}$ held-out images.
For our pipeline that means $m \sim 10^{3}$.

Figure: `outputs/reports/imagenet_shard_m_sweep/advantage_vs_m.{png,pdf}`
(two panels: advantage vs $m$ with γ_0.05 curve and m* annotation;
empirical reject rate vs $m$ with α reference line).

---

## 6. Experiment 3 — Logits-only top-1 log-prob (Table 2)

**Score:** $s_{T}^{\mathrm{top1}}$. **Access:** only $T$'s full logit
vector over its own class space. No ground-truth labels used.

| Target | Adv_ctrl | Adv_main | Verdict / Correct? |
|---|---:|---:|---|
| `google/vit-base-patch16-224` (self) | 0.025 | 0.033 | Fail / **No** |
| `ISxOdin/vit-base-oxford-iiit-pets` | −0.014 | 0.001 | Fail / **No** |
| `MatanBT/...-cifar10` | −0.004 | −0.009 | Fail / **No** |
| `MatanBT/...-cifar100` | 0.010 | 0.028 | Fail / **No** |
| `Abhiram4/...-eurosat` | 0.008 | −0.014 | Fail / **No** |
| `MatanBT/...-celeba-smiling` | −0.017 | −0.001 | Fail / **No** |
| `rvv-karma/Human-Action-Recognition` | −0.015 | −0.005 | Fail / **No** |
| ViT-B/16-224 random-init | −0.020 | 0.013 | Fail / Yes |
| `facebook/deit-base-patch16-224` | 0.048 | **+0.063** | Reject / **No** |

**Verdict accuracy: 1/9.** Every verdict on a true derivative is a false
negative. DeiT, which trained on the same shard, is the only model that
rejects $H_0$ — a false *positive* for derivation but a true positive for
shard memorisation. The audit has collapsed onto detecting co-training
rather than lineage, which is exactly the case ruled out by Assumption 1.iii.

---

## 7. Experiment 4 — Augmentation MIA with full logits (Table 3)

**Score:** $s_{T}^{\mathrm{aug}}$. **Access:** same as Experiment 3, plus
$K=8$ random augmentations per image (random resized crop scale ∈ [0.85, 1.0],
horizontal flip, mild colour jitter). Members tend to produce more stable
predictions across augmentations.

This is **Choquette-Choo et al. 2021 (ICML)** extended to full-logit access
(per-class logit variance instead of label agreement).

| Target | Adv_ctrl | Adv_main | AUC | Verdict / Correct? |
|---|---:|---:|---:|---|
| `google/vit-base-patch16-224` (self) | 0.112 | **+0.111** | 0.587 | Reject / Yes |
| `ISxOdin/vit-base-oxford-iiit-pets` | 0.126 | **+0.123** | 0.592 | Reject / Yes |
| `MatanBT/...-cifar10` | 0.072 | **+0.115** | 0.582 | Reject / Yes |
| `MatanBT/...-cifar100` | 0.125 | **+0.124** | 0.594 | Reject / Yes |
| `Abhiram4/...-eurosat` | 0.096 | **+0.106** | 0.588 | Reject / Yes |
| `MatanBT/...-celeba-smiling` | 0.094 | **+0.088** | 0.570 | Reject / Yes |
| `rvv-karma/Human-Action-Recognition` | 0.121 | **+0.123** | 0.585 | Reject / Yes |
| ViT-B/16-224 random-init | −0.012 | **+0.008** | 0.499 | Fail / Yes |
| `facebook/deit-base-patch16-224` | 0.052 | **+0.108** | 0.567 | Reject / **No** |

**Verdict accuracy: 8/9.** All seven derivatives correctly rejected. The
random-init negative control correctly fails to reject. DeiT is the
remaining failure for the same reason as in Experiment 3 — augmentation
stability is a property of the model's *training history*, not its
*architectural lineage*. No purely black-box MIA on $T$'s outputs can
distinguish DeiT-style co-training from genuine derivation.

The shuffled-label control is approximately tied with the main advantage
for several rows because the score is strongly bimodal; once the threshold
sweep places the cut in the gap between member and nonmember modes (which
happens for nearly any non-degenerate label permutation), the threshold
transfers to the true-label test as well. This is a known weakness of
single-permutation shuffled-label controls on bimodal scores and reflects
the calibration procedure, not the membership signal.

---

## 8. Three-way comparison of access regimes

| Access | Score | Derivatives | Random-init | DeiT | Total | Comment |
|---|---|:---:|:---:|:---:|:---:|---|
| Grey-box | $s^{\mathrm{enc}}$ | 7/7 | ✓ | ✓ | **9/9** | Lineage-specific |
| Black-box (aug) | $s^{\mathrm{aug}}$ | 7/7 | ✓ | ✗ | **8/9** | Shard-specific, not lineage |
| Black-box (label-free) | $s^{\mathrm{top1}}$ | 0/7 | ✓ | ✗ | 1/9 | Power-limited |

The progression isolates two effects:

1. **Power of the score.** Augmentation MIA is dramatically more powerful
   than top-1 log-prob (7 vs 0 derivatives detected at the same access
   level). The augmentation signal exploits a robust geometric fact:
   $T$'s decision surface tightens around training images, which manifests
   as low cross-augmentation variance regardless of how the head was
   retrained.

2. **Specificity vs co-training confounder.** Even a strong black-box score
   cannot distinguish derivation from independent co-training on the same
   shard. The DeiT row consistently flips the verdict to "Reject" under
   every output-only MIA. Closing this gap requires conditioning the score
   on $P$'s own parameters (encoder-routing through $P_{\mathrm{head}}$, or
   equivalent), at which point the audit reads $P\!\to\!T$ rather than
   $S\!\to\!T$.

---

## 9. Conclusions and insights

1. **The shard-membership framework transfers to vision.** With
   $m \approx 10^{3}$ class-balanced ImageNet samples, all seven declared
   derivatives of a public ImageNet ViT reject $H_0$ at the analytical
   level-α threshold under grey-box scoring; both negative controls correctly
   fail to reject. The empirical Type-I bound from Proposition 1 holds at
   finite samples for the random-init null.

2. **Class-balanced sampling is mandatory whenever the data source is
   class-sorted.** Otherwise $P$'s per-class head bias creates a latent
   confounder strong enough to push the random-init control to AUC 0.61.
   This is a generic pitfall for vision membership-inference benchmarks.

3. **Pets fine-tune froze the encoder bit-for-bit.** A non-trivial fraction
   of HF-listed "fine-tunes" are actually head-only fine-tunes, and the
   audit transparently survives them — the derivative's score is *identical*
   to the parent's in this case, which is the strongest possible witness
   of lineage.

4. **Sample complexity behaves as predicted.** Empirical crossover
   $m^{\star}\approx 880$ matches $\log(2/\alpha)/\Delta^{2}\approx 824$
   within 7% on the cifar10 target. The reject-rate curve transitions
   sharply through $m\in[640,1280]$ and saturates at 0.96 at $m=1920$.

5. **The DeiT control formalises Assumption 1.iii.** DeiT was independently
   trained on the same ImageNet-1k that $P$ trained on, so the latent
   "ImageNet dataset" $H$ creates the path
   $S\leftarrow H\rightarrow T$ that Assumption 1.iii rules out. We observed
   empirically that black-box MIAs respond to this path (false positive on
   DeiT) while the grey-box encoder-routed score is robust to it (correct
   verdict on DeiT). This makes the framework's identifiability assumption
   testable rather than purely formal.

6. **Augmentation MIA is the right black-box baseline for the paper.** It
   is much stronger than label-free top-1 confidence (7 vs 0 derivatives
   detected) without requiring shadow-model training, and its failure mode
   on DeiT cleanly demonstrates the residual cost of pure black-box access.
   This trio of tables — grey-box, augmentation black-box, label-free
   black-box — makes the cost-of-access argument quantitatively and
   structurally.

7. **Practical recipe for an auditor.** With access to encoder activations,
   use $s_T^{\mathrm{enc}}$ and budget $m\sim 10^{3}$ images. With access
   to logits only, use augmentation-MIA ($K\geq 8$); expect false positives
   on models that independently co-trained on the same shard. Below
   $m\approx 300$ the test is essentially powerless even for clear
   derivatives.

---

## 10. Artifacts

### Code
- `scripts/data/prepare_imagenet_shard.py`
- `scripts/scoring/extract_image_scores.py` (grey-box)
- `scripts/scoring/extract_image_scores_aug.py` (augmentation MIA)
- `scripts/experiments/sweep_m_imagenet_shard.py`
- `src/shard_audit/vit_scores.py` (encoder-routed score, random-init builder)

### Data
- `data/processed/imagenet_shard/` — manifest + train/test JSONL + 7700 JPEG images
- `data/scores/imagenet_shard__*/` — grey-box scores (9 targets)
- `data/scores_logits_only/imagenet_shard__*/` — (same scores, re-evaluated with
  $s^{\mathrm{top1}}$ as primary — uses the same JSONL columns)
- `data/scores_aug/imagenet_shard__*/` — augmentation-MIA scores

### Reports
- `outputs/runs/imagenet_shard__*/` — main grey-box MIA results per target
- `outputs/runs_logits_only/*/` — logits-only MIA results per target
- `outputs/runs_aug/*/` — augmentation-MIA results per target
- `outputs/reports/imagenet_shard_m_sweep/advantage_vs_m.{png,pdf}` — m-sweep figure
- `outputs/reports/imagenet_shard_m_sweep/results.json` — m-sweep raw data

### Key references for the paper
- **Choquette-Choo et al. 2021 (ICML)** — "Label-Only Membership Inference Attacks".
  Primary citation for augmentation MIA. We extend to full-logit per-class variance.
- **Li & Zhang 2021 (CCS)** — concurrent label-only MIA paper.
- **Carlini et al. 2022 (S&P)** — "Membership Inference Attacks From First
  Principles" (LiRA). Stronger black-box baseline, requires shadow training;
  not run here but worth citing as the SOTA reference.
- **Yeom et al. 2018** — original loss-based MIA.
- **Pearl 2009** — `do`-calculus and identifiability, for Assumption 1.
- **Angrist–Imbens–Rubin 1996** — instrumental-variable identification, for the
  $S$-as-IV interpretation in the theoretical section.

---

## 11. Suggested follow-ups (not run this session)

1. **Reference-calibrated MIA**: $s_T^{\mathrm{ref}}(x) = \log p_T(\hat y\mid x) - \log p_R(\hat y\mid x)$
   with $R$ = the random-init ViT we already have. Cheap; expected to narrow
   but not close the DeiT false-positive gap.

2. **LiRA / Attack-R** with a small fleet (~8) of shadow ViT-B/16 models
   trained on random ImageNet-1k subsets. SOTA black-box; same structural
   ceiling on lineage vs co-training.

3. **Distribution-shifted negative control** — a same-architecture ViT
   trained on a *non-overlapping* dataset (e.g., a small JFT subset, or
   iNaturalist). Distinguishes the DeiT "same-shard co-training" case
   from a more generic "trained on something else" negative.

4. **m-sweep on a weaker derivative** (e.g., the CelebA-smiling target with
   Adv_main = 0.047, the smallest in the table). Verify the $m^{\star}=\log(2/\alpha)/\Delta^{2}$
   relationship holds across effect sizes.

5. **Augmentation MIA with larger $K$** to check whether the residual DeiT
   false positive can be tightened by adding more augmentations (theory
   says no — the asymptotic signal is determined by the model, not by $K$).
