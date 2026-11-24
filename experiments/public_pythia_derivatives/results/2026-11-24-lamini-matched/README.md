# qmGv.Q1(b) · Matched Lamini Docs experiment

**Yes, the public MIMIR GitHub audit fired on BioGPT-Large after downstream adaptation.** BioGPT’s primary MIN-K20 advantage increased from +0.010 to +0.275. Pythia retained a positive signal, moving from +0.380 to +0.330. Both final checkpoints also crossed the prespecified cutoff for the exploratory mean log-probability score.

## Held-out MIMIR results

Each of the four checkpoints used the same 500 construction and 200 evaluation snippets per source. A positive decision classifies a snippet as coming from the Pile member source. Advantage is the difference between member- and nonmember-source acceptance rates. Each checkpoint and score has its own construction-fitted threshold. The single-test cutoff was 0.135810. At α = 0.05 the two adapted models formed the primary MIN-K20 family with a Bonferroni cutoff of 0.148021. The two mean-score tests formed a separate exploratory family with the same cutoff. The base rows are descriptive. The shuffle column uses the prespecified seed-42 label permutation, including refitting the threshold.

| Checkpoint | Score | Fitted threshold | Accepted member / nonmember (of 200 each) | Signed gap | Absolute gap | Shuffled gap | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| Pythia-1.4B base | MIN-K20 | −6.914857 | 125 / 49 | +0.380 | 0.380 | −0.035 | Above individual cutoff |
| Pythia-1.4B base | Mean log probability | −2.134361 | 129 / 51 | +0.390 | 0.390 | +0.010 | Above individual cutoff |
| Pythia-1.4B + Lamini | MIN-K20 | −7.359026 | 113 / 47 | +0.330 | 0.330 | −0.080 | Detects after correction |
| Pythia-1.4B + Lamini | Mean log probability | −2.143917 | 106 / 31 | +0.375 | 0.375 | −0.020 | Detects after correction |
| BioGPT-Large base | MIN-K20 | −16.132413 | 198 / 196 | +0.010 | 0.010 | −0.040 | Below individual cutoff |
| BioGPT-Large base | Mean log probability | −6.626948 | 181 / 181 | 0.000 | 0.000 | −0.040 | Below individual cutoff |
| BioGPT-Large + Lamini | MIN-K20 | −11.304104 | 121 / 66 | +0.275 | 0.275 | +0.040 | Detects after correction |
| BioGPT-Large + Lamini | Mean log probability | −5.324074 | 136 / 101 | +0.175 | 0.175 | −0.065 | Detects after correction |

The BioGPT score rankings also changed. Held-out AUC went from 0.404 to 0.672 for MIN-K20 and from 0.443 to 0.570 for mean log probability. Thus the measured change in score separation is also visible independently of the fitted thresholds. [Full-precision witness results](witness_audit.csv)

### Paired changes in signed advantage

The intervals are percentile intervals from 10,000 source-stratified paired bootstrap draws on the held-out snippets, shared across models and scores. They condition on the available evaluation pools and frozen fitted rules.

| Contrast | MIN-K20 change [95% interval] | Mean log-probability change [95% interval] |
|---|---:|---:|
| Adapted Pythia minus its base | −0.050 [−0.100, 0.000] | −0.015 [−0.075, +0.045] |
| Adapted BioGPT minus its base | +0.265 [+0.170, +0.360] | +0.175 [+0.085, +0.270] |
| Adapted Pythia minus adapted BioGPT | +0.055 [−0.055, +0.165] | +0.200 [+0.100, +0.300] |

[Paired estimates](witness_changes.csv)

## Did both models adapt to Lamini?

Both ran rank-16 LoRA for three complete epochs on identical row permutations, 1,260 training examples per epoch and 237 optimizer updates per model. Native tokenizers yielded 80,204 supervised answer/EOS tokens per Pythia epoch and 80,699 per BioGPT epoch. Every training row fit at 512 tokens. We evaluated the fixed final checkpoints on all 140 official test rows. On the 124 test rows whose question and answer were individually unseen in training, loss also fell in both models.

| Model | Full train NLL, before → after | Full test NLL, before → after | Test NLL reduction | Test perplexity, before → after | Test NLL change [paired 95% interval] | 124-row NLL, before → after |
|---|---:|---:|---:|---:|---:|---:|
| Pythia-1.4B | 2.085 → 1.297 | 1.960 → 1.353 | 30.9% | 7.10 → 3.87 | −0.606 [−0.668, −0.552] | 1.904 → 1.357 |
| BioGPT-Large | 2.840 → 1.869 | 2.710 → 1.884 | 30.5% | 15.03 → 6.58 | −0.826 [−0.906, −0.756] | 2.610 → 1.855 |

Loss is token-weighted negative log likelihood on the answer and EOS. The intervals come from 10,000 paired resamples of the official test rows. NLLs are comparable within each native tokenizer. [Downstream metrics](downstream.csv) · [Training verification](../../../../artifacts/lamini_matched/training_verification.json)

## Rebuttal paragraph

Yes. We fine-tuned Pythia-1.4B and independently developed, PubMed-pretrained BioGPT-Large on all 1,260 examples of the same Lamini Docs training split. Both improved held-out downstream answer loss by about 30%. On 200 MIMIR GitHub evaluation examples per source, the MIN-K20 advantage was 0.330 for adapted Pythia and 0.275 for adapted BioGPT, compared with 0.380 and 0.010 before adaptation. Both exceeded the two-target Bonferroni cutoff of 0.1480. The corresponding shuffled-label gaps were −0.080 and +0.040. Mean log-probability advantages also crossed their exploratory corrected cutoff. Thus a model with no Pythia lineage and no documented GitHub pretraining acquired a detectable public-shard signal after Lamini adaptation. The experiment leaves open which features of fine-tuning and of the train–validation split produced that signal.

## Scope and reproducibility

The prespecified dataset check found zero full-snippet matches among the 1,400 GitHub records, and zero contiguous 13-word matches among the 1,398 long enough for comparison. Microsoft documents PubMed pretraining for BioGPT-Large. Its exact pretraining document manifest is unavailable. The single matched training run per family cannot distinguish a Lamini-specific effect from a more general effect of adaptation. Formal error control on fresh public shards also remains unverified. [Dataset and pretraining verification](../2026-11-24-lamini-design/README.md) · [Pinned BioGPT model card](https://huggingface.co/microsoft/BioGPT-Large/blob/c6a5136a91c5e3150d9f05ab9d33927a3210a22e/README.md)

The GPU phase ran directly on one Vast.ai A100-SXM4-40GB for 13 minutes 2 seconds, an allocated rental cost of about $0.10 at $0.46778/hour. The training loops took 193.3 seconds for Pythia and 377.6 seconds for BioGPT. Peak memory reserved by PyTorch during training was 5.08 and 5.30 GiB, respectively. Pythia’s preliminary in-memory zero-adapter check differed from the cached FP16 scores. We kept its trained adapter, serialized and reloaded the FP16 export using the audit’s actual loading path, and repeated the check. Both reloaded zero-update exports reproduced the cached scores on all four fixed construction records before scoring the corresponding adapted export. Both adapted models used their pinned native tokenizer, and all four score caches passed row-identity checks. [Timing and memory](compute.json) · [Training and audit protocol](../2026-11-24-lamini-design/protocol.yaml) · [Replay instructions](../../../../artifacts/lamini_matched/REPRODUCE.md)

[Pythia adapter](../../../../artifacts/lamini_matched/pythia/adapter) · [BioGPT adapter](../../../../artifacts/lamini_matched/biogpt/adapter) · [Pythia merged FP16 checkpoint](../../../../artifacts/lamini_matched/pythia/merged) · [BioGPT merged FP16 checkpoint](../../../../artifacts/lamini_matched/biogpt/merged) · [Portable archive with both adapters, inputs, per-row results and code](../../../../artifacts/lamini_matched.zip)
