# qmGv.Q3 · Public MIMIR audit on BioGPT-Large

On the current MIMIR GitHub split, the two fitted BioGPT-Large rules gave held-out advantages of +0.010 and 0.000. Neither crossed the 0.13581 per-score reference cutoff. BioGPT-Large provides a trained control with no documented GitHub pretraining. Pythia step 0 provides a reference before weight training. The comparison addresses how often the public audit rejects on these particular shards.

## Held-out decisions

Each positive decision classifies a text as coming from the Pile member source. Signed advantage is the member-source acceptance rate minus the nonmember-source acceptance rate. At 200 texts per source and α = 0.05, all four comparisons fell below the cutoff.

| Model | Score | Fitted score threshold | Member accepted | Nonmember accepted | Signed advantage | Absolute advantage | Raw-score AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| BioGPT-Large | MIN-K20 | −16.132413 | 198/200 | 196/200 | +0.010 | 0.010 | 0.404 |
| BioGPT-Large | Mean log probability | −6.626948 | 181/200 | 181/200 | 0.000 | 0.000 | 0.443 |
| Pythia-1.4B, initialization | MIN-K20 | −11.714882 | 41/200 | 29/200 | +0.060 | 0.060 | 0.565 |
| Pythia-1.4B, initialization | Mean log probability | −10.992433 | 98/200 | 89/200 | +0.045 | 0.045 | 0.559 |

BioGPT's scores tended to rank nonmember-source snippets higher than member-source snippets. The prespecified rule treats higher scores as more member-like. Its MIN-K20 threshold classified almost every snippet as a member, which explains how an AUC of 0.404 coexists with a +0.010 threshold advantage. The source differences are visible in tokenization as well. The median BioGPT input length was 66 tokens on the held-out member source and 80 on the nonmember source. The corresponding Pythia medians were 51 and 63. Each tokenizer truncated one nonmember evaluation text at the 512-token cap.

## Rejection frequencies

For each replicate we sampled separately and without replacement from each of the two fixed evaluation pools. Both models and scores saw the same selections. We kept the original construction thresholds fixed. A replicate rejected when the absolute advantage strictly exceeded √(log(2/α)/m), where m is the number of examples selected per source. Each of the 10,000 replicates used one independent random permutation per source, with the first m entries determining the subset at each size. The simulation used seed 3324.

At α = 0.05, the observed frequencies were:

| Evaluation texts per source | BioGPT MIN-K20 | BioGPT mean | Random Pythia MIN-K20 | Random Pythia mean |
|---:|---:|---:|---:|---:|
| 25 | 0% | 0% | 0.07% | 0.60% |
| 50 | 0% | 0% | 0.07% | 0.45% |
| 100 | 0% | 0% | 0.03% | 0.13% |
| 150 | 0% | 0% | 0% | 0.01% |
| 200 | 0% | 0% | 0% | 0% |

Both BioGPT scores yielded 0/10,000 rejections in each of the 20 combinations of α ∈ {0.01, 0.025, 0.05, 0.10} and m ∈ {25, 50, 100, 150, 200}. The largest observed random-Pythia frequencies were 0.32% for MIN-K20 and 1.40% for mean log probability, both at α = 0.10 and m = 25. At m = 200 every replicate evaluates the entire pool and has the same outcome.

We also computed the exact probability of rejection conditional on each pair of held-out acceptance counts. BioGPT MIN-K20 can differ by at most four accepted examples between two equal-size subsets, which makes crossing impossible throughout this grid. BioGPT mean log probability can cross for some subsets. Its largest exact probability was 0.001318% (α = 0.10, m = 25). Conditioning on one pair of public shards and a fixed fitted rule measures sensitivity to the choice of evaluation texts. Population type-I error under new train–validation splits remains unmeasured. [All 80 cells and exact probabilities](subsampling.md) · [Subsampling CSV](subsampling.csv)

## Corpus and witness overlap

Microsoft lists PubMed for the pretrained BioGPT-Large checkpoint and publishes a separate PubMedQA-finetuned checkpoint. The BioGPT paper describes training on roughly 15 million PubMed titles and abstracts. There is no published document-level manifest establishing the exact contents of the Large checkpoint's training corpus. We therefore use “no documented GitHub training exposure.” [Pinned Microsoft model card](https://huggingface.co/microsoft/BioGPT-Large/blob/c6a5136a91c5e3150d9f05ab9d33927a3210a22e/README.md) · [Microsoft release table](https://github.com/microsoft/BioGPT#pre-trained-models) · [Luo et al.](https://doi.org/10.1093/bib/bbac409)

We screened all 1,400 audit snippets against 75,664 distinct PubMed titles and abstracts with pre-2021 entry dates from three NLM baseline files. No full normalized snippet matched. None of the 1,398 snippets with at least 13 normalized words shared a contiguous 13-word passage with the screened records. The other two were included in the full-text comparison. This was a targeted subset of PubMed using later versions of historically indexed records. Incidental overlap elsewhere, shorter quotations and paraphrases remain possible. [Screening methods and NLM file identities](../../../../artifacts/biogpt_large_mimir/corpus.md)

## Run configuration and reproduction

We used the supplied MIMIR GitHub files, configuration `ngram_13_0.2`, seed 0. They contain 500 snippets per source for construction and 200 per source for evaluation. All 1,400 snippets are distinct and 32 whitespace-delimited words long. Both models' score records matched the input row identities and labels. The historical Table 6 used a different evaluation split. We ran BioGPT-Large directly on one Vast.ai NVIDIA A100-SXM4-40GB and reused the same-split Pythia initialization scores. Each model used its own pinned tokenizer. [MIMIR data manifest](../../../../data/processed/mimir_github/manifest.json) · [BioGPT-Large checkpoint](https://huggingface.co/microsoft/BioGPT-Large/tree/c6a5136a91c5e3150d9f05ab9d33927a3210a22e) · [Pythia step-0 checkpoint](https://huggingface.co/EleutherAI/pythia-1.4b/tree/ae04bdc1923b877cf65aee9da264d30555f011ab)

The scorer used unquantized FP16 inference, batches of four texts and a 512-token cap. It scored observed next tokens, excluding padding. Mean log probability averages all scored tokens. MIN-K20 averages the lowest ceil(0.2 × number of scored tokens), with a minimum of one. The repository audit fitted each score threshold by maximizing balanced accuracy on the construction split over 2,001 ascending grid points, taking the first maximizer. All frozen decisions were independently recovered from the score caches. [Full-precision results](advantages.csv) · [Paired subsampling script](../../reports/subsample_control_scores.py)

The run used Transformers 4.44.2, PyTorch 2.14.1+cu132, Accelerate 1.14.0, Sacremoses 0.2.0, NumPy 1.26.4, SciPy 1.17.1 and scikit-learn 1.9.0. We directed the repository loader to place BioGPT on the explicit device cuda:0. The reproduction bundle includes the prepared inputs, both score caches, fitted audits, source code used by the runs and the overlap checker. [Reproduction bundle](../../../../artifacts/biogpt_large_mimir.zip)

## Rebuttal paragraph

The level-α guarantee does not generally extend to public train-versus-validation shards. To measure the practical effect of this shift on MIMIR GitHub, we evaluated BioGPT-Large, whose documented pretraining source is PubMed. A screen of 75,664 PubMed titles and abstracts found no full witness match. With 200 held-out examples per source, BioGPT's MIN-K20 and mean log-probability advantages were 0.010 and 0.000, compared with 0.060 and 0.045 at Pythia initialization. All four were below the 0.1358 cutoff at α = 0.05. BioGPT produced no rejections in 10,000 subsamples per setting across four significance levels and five evaluation sizes. Thus the observed source differences seldom triggered these fitted audits on the tested controls. This evidence is conditional on the available shards and thresholds. Incidental training overlap and the false-positive rate on fresh public splits remain unresolved.
