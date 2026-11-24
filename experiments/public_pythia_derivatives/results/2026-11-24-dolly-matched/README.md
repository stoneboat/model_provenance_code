# Dolly-15K matched downstream experiment

**Yes, the MIMIR GitHub audit fired on BioGPT-Large after Dolly fine-tuning.** BioGPT’s primary MIN-K20 advantage went from +0.010 to +0.250, a change of +0.240 (paired 95% interval +0.145 to +0.335). Its mean log-probability advantage went from 0.000 to +0.240. Pythia retained its positive signal with gains of +0.005 and +0.020 over its pretrained baseline. The two final checkpoints were fine-tuned separately from their original pretrained bases. Each saw every one of the 15,011 Dolly examples once, in the same seed-42 row order, for 939 optimizer updates. The last update contained three examples.

## MIMIR GitHub witness audit

| Checkpoint | Score | Fitted threshold | Member / nonmember accepted (200 each) | Signed advantage | Absolute advantage | AUC | Shuffled gap | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pretrained Pythia-1.4B | MIN-K20 | -6.914857 | 125 / 49 | +0.380 | 0.380 | 0.736 | -0.035 | Descriptive base |
| Pretrained Pythia-1.4B | Mean log probability | -2.134361 | 129 / 51 | +0.390 | 0.390 | 0.753 | +0.010 | Descriptive base |
| Pythia-1.4B + Dolly | MIN-K20 | -6.872723 | 126 / 49 | +0.385 | 0.385 | 0.739 | -0.015 | Detects after correction |
| Pythia-1.4B + Dolly | Mean log probability | -1.854076 | 113 / 31 | +0.410 | 0.410 | 0.753 | 0.000 | Detects after correction |
| Pretrained BioGPT-Large | MIN-K20 | -16.132413 | 198 / 196 | +0.010 | 0.010 | 0.404 | -0.040 | Descriptive base |
| Pretrained BioGPT-Large | Mean log probability | -6.626948 | 181 / 181 | 0.000 | 0.000 | 0.443 | -0.040 | Descriptive base |
| BioGPT-Large + Dolly | MIN-K20 | -10.363840 | 106 / 56 | +0.250 | 0.250 | 0.659 | +0.050 | Detects after correction |
| BioGPT-Large + Dolly | Mean log probability | -4.647956 | 128 / 80 | +0.240 | 0.240 | 0.613 | +0.030 | Detects after correction |

Construction used the same 500 member-source and 500 nonmember-source texts as Lamini. Each model and score received its own threshold, fitted by maximizing construction balanced accuracy. Evaluation used the same frozen 200 texts per source. Signed advantage is the member-source acceptance rate minus the nonmember-source acceptance rate. Higher scores are more member-like. A positive decision requires the signed gap to strictly exceed the cutoff. The individual cutoff is 0.135810 at α = 0.05. The two adapted models form the primary MIN-K20 family, with a Bonferroni cutoff of 0.148021. The two mean-score tests form a separate exploratory family. The shuffled control refits the threshold after a seed-42 permutation of the construction labels and evaluates on separately shuffled evaluation labels. Both pretrained rows reuse the original score caches. [Full numerical results](witness_audit.csv)

## Paired changes

| Contrast | Score | Difference in signed advantage | 95% interval |
| --- | --- | --- | --- |
| Pythia adaptation minus base | MIN-K20 | +0.005 | [-0.010, +0.025] |
| Pythia adaptation minus base | Mean log probability | +0.020 | [-0.035, +0.075] |
| BioGPT adaptation minus base | MIN-K20 | +0.240 | [+0.145, +0.335] |
| BioGPT adaptation minus base | Mean log probability | +0.240 | [+0.140, +0.335] |
| Adapted Pythia minus adapted BioGPT | MIN-K20 | +0.135 | [+0.020, +0.250] |
| Adapted Pythia minus adapted BioGPT | Mean log probability | +0.170 | [+0.065, +0.280] |

The intervals use 10,000 paired, source-stratified bootstrap draws with seed 3324 and condition on the two evaluation pools and fitted thresholds. [Paired comparisons](witness_changes.csv)

## Comparison with Lamini

| Model family | Score | After Lamini | After Dolly | Dolly minus Lamini | Paired 95% interval |
| --- | --- | --- | --- | --- | --- |
| Pythia-1.4B | MIN-K20 | +0.330 | +0.385 | +0.055 | [+0.005, +0.105] |
| Pythia-1.4B | Mean log probability | +0.375 | +0.410 | +0.035 | [-0.010, +0.080] |
| BioGPT-Large | MIN-K20 | +0.275 | +0.250 | -0.025 | [-0.095, +0.045] |
| BioGPT-Large | Mean log probability | +0.175 | +0.240 | +0.065 | [0.000, +0.130] |

Both studies started from the same base checkpoints and used the same native tokenizers, LoRA settings and audit. Lamini used three epochs on 1,260 examples (237 updates). Dolly used one epoch on 15,011 examples (939 updates). Both the corpus and the optimization exposure differ across these runs. [Comparison data](dolly_vs_lamini.csv)

## Adaptation to Dolly

| Checkpoint | Base response NLL | Adapted response NLL | Reduction | Perplexity before → after | Supervised tokens |
| --- | --- | --- | --- | --- | --- |
| Pythia-1.4B + Dolly | 2.122 | 1.967 | 7.3% | 8.35 → 7.15 | 1,142,470 |
| BioGPT-Large + Dolly | 2.993 | 2.577 | 13.9% | 19.95 → 13.16 | 1,212,081 |

Dolly contains one official training split. The loss above is next-token negative log likelihood on response and EOS tokens for all training rows, with padding and prompt labels masked. No separate Dolly evaluation set was constructed. The MIMIR evaluation rows remained held out throughout fitting. [Downstream metrics](downstream.csv)

## Rebuttal paragraph

We repeated the matched downstream experiment using one LoRA epoch on all 15,011 Dolly examples. On the frozen MIMIR GitHub evaluation shards (200 examples per source), Pythia’s MIN-K20 advantage was 0.380 before and 0.385 after fine-tuning. PubMed-pretrained BioGPT-Large went from 0.010 to 0.250. Both adapted models exceeded the two-target Bonferroni cutoff of 0.1480, while their shuffled-label gaps were −0.015 and +0.050. Mean log-probability advantages also crossed the separately corrected exploratory cutoff. Together with the Lamini experiment, this shows that the public audit can fire in an independently developed model after adaptation on either of two downstream datasets. Which features of adaptation made the fixed source shards distinguishable remains unresolved.

## Inputs, verification and compute

The release was pinned to bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a and includes 4,467 nonempty context fields. Each row used its instruction, optional context and response. The 512-token limit shortened 878 Pythia rows and 1,031 BioGPT rows. The overflow rule removes optional context first, retains instruction and response content, and terminates the supervised sequence with EOS. Every row contributed real instruction and response tokens in both models. A local screen against the entire raw downstream dataset found no full normalized MIMIR snippet match among 1,400 snippets and no contiguous 13-word match among the 1,398 eligible snippets. Shorter quotations and paraphrases remain possible. [Databricks dataset, CC BY-SA 3.0](https://huggingface.co/datasets/databricks/databricks-dolly-15k/tree/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a) · [Truncation and overlap records](../../../../artifacts/dolly_matched/data) · [Exact protocol](../../configs/matched_dolly.yaml)

The pretrained checkpoints were EleutherAI/pythia-1.4b at fedc38a16eea3bd36a96b906d78d11d2ce18ed79 and microsoft/BioGPT-Large at c6a5136a91c5e3150d9f05ab9d33927a3210a22e. Native tokenizers and unquantized FP16 MIMIR inference matched the earlier experiment. Both zero-update FP16 exports reproduced both cached base scores on the four fixed construction texts. The training verifier checked every optimizer group against its original row indices and actual supervised token counts. Dollar cost is the allocation for the GPU pipeline on the existing rental.

| Resource | Actual |
|---|---:|
| GPU | 1 × Vast A100-SXM4-40GB |
| Pythia training loop | 12.8 min |
| BioGPT training loop | 25.0 min |
| Elapsed GPU pipeline including full response-loss evaluations | 51 min 4 s |
| Allocated rental at $0.46778/h | $0.40 |

[Compute measurements](compute.json) · [Training verification](../../../../artifacts/dolly_matched/training_verification.json) · [Reproduction instructions](../../../../artifacts/dolly_matched/REPRODUCE.md) · [Portable bundle including both adapters](../../../../artifacts/dolly_matched.zip)
