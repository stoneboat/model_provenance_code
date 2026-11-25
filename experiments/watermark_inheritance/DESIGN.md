# Pythia-1B watermark design

Section 5.1 uses a frozen Pythia-1B generator to draw two independent
continuations per WikiText prompt with the same key and generation settings.
The witness \(S\) contains one randomly assigned completion per prompt.
The other forms \(S'\). The parent is fine-tuned on \(S\). The descendant
starts downstream training from the parent and the control starts from the
unmarked base.

## Pinned inputs

| Input | ID and revision |
| --- | --- |
| Generator, base weights and tokenizer | `EleutherAI/pythia-1b` at `f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2` |
| Prompts | `Salesforce/wikitext` at `b08601e04326c79dfdd32d625aee71d232d685c3`, `wikitext-103-raw-v1/train` |
| Dolly | `databricks/databricks-dolly-15k` at `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`, `train` |
| Watermark code | `facebookresearch/radioactive-watermark` at `19c6dc90e7a0dddb1a241b73e7fae708b68f024e` |

The [configuration](configure.py) fixes the three model seeds, watermark
keys, and generation settings. Prompt preparation chooses distinct 64-token
prefixes, one per article. The Maryland watermark uses context width 2,
greenlist fraction 0.25, logit bias 3, temperature 0.8, top-p 0.95, and up
to 256 new tokens. Completion seeds and assignment seeds are separate.
Identical sampled pairs are retained.

For the upstream compatibility tests, clone the pinned code from the
repository root. The Slurm setup job also prepares this checkout.

```bash
mkdir -p .cache/watermark
git clone https://github.com/facebookresearch/radioactive-watermark .cache/watermark/radioactive-watermark
git -C .cache/watermark/radioactive-watermark checkout --detach 19c6dc90e7a0dddb1a241b73e7fae708b68f024e
```

## Train and audit

The parent uses all 10,000 witness completions. Each Dolly stage uses the
same response-only formatting and checks for overlap with the generated
shards. Full-parameter AdamW uses learning rate \(10^{-5}\), an effective
batch size of 32, cosine decay with 3% warmup, a 1,024-token limit, and
three epochs. The downstream stage starts with a fresh optimizer.
[Training implementation](train.py) · [Archived input and run settings](results/2026-10-06/)

The reading-mode scorer uses the checkpoint's highest-probability
next token under each stored completion prefix. It computes the
eligible green-token score
\(z=(G-N\gamma)/\sqrt{N\gamma(1-\gamma)}\), with vocabulary-rounded
\(\gamma\), repetition exclusions within each example, and neutral score
for zero eligible positions. It fits a threshold by balanced accuracy on
2,000 prompt pairs, then uses a strict score comparison on 8,000 different
pairs. These 8,000 pairs are held out from detector calibration. The
parent's training includes their \(S\) completions.

The paper reports the signed difference \(a-a'\) in held-out acceptance
rates. Rejection uses \(|a-a'|>\sqrt{2\log(2/\alpha)/m}\) for the paired
design. At \(\alpha=0.05\) and \(m=8,000\), the cutoff is 0.0304. The scorer
also stores a paired-swap p-value and diagnostic scores.
[Scoring implementation](score.py) · [Full run commands and tables](README.md)
