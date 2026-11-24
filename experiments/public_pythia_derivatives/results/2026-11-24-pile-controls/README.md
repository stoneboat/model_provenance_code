# Pile controls on the current MIMIR GitHub split

Completed directly on one Vast.ai A100 on November 24, 2026. The public shard signal crossed the reference cutoff in the independent GPT-Neo model as well as in the trained Pythia parent. Tulu-v2-mix was the only tested designated derivative below the cutoff with both scores.

All eight checkpoints used the same 32-word snippets, with 500 examples per source for fitting and 200 per source for evaluation. Each score threshold was fitted separately on the construction split by maximizing balanced accuracy. Held-out advantage is the member-source acceptance rate minus the nonmember-source acceptance rate. All signed advantages were positive, so the reported values also give their absolute magnitudes. We applied the strict two-sided reference rule at α = 0.05 to each score, with cutoff 0.13581015.

| Model | MIN-K20 threshold | MIN-K20 advantage | Mean log probability threshold | Mean log probability advantage | Both cross 0.13581? |
|---|---:|---:|---:|---:|---|
| Pythia-1.4B, initialization | -11.7149 | +0.060 | -10.9924 | +0.045 | No |
| GPT-Neo-1.3B, independent Pile model | -6.5981 | +0.305 | -1.9295 | +0.355 | Yes |
| Pythia-1.4B, trained parent | -6.9149 | +0.380 | -2.1344 | +0.390 | Yes |
| Lamini Docs | -8.0429 | +0.345 | -2.2395 | +0.365 | Yes |
| Tulu-v2-mix | -11.5358 | +0.105 | -5.9414 | +0.080 | No |
| Fin-Pythia | -13.1257 | +0.170 | -4.2686 | +0.240 | Yes |
| Helpful DPO | -9.0774 | +0.290 | -2.4915 | +0.355 | Yes |
| Helpful SFT | -7.8892 | +0.315 | -2.2372 | +0.365 | Yes |

The controls leave the relative contributions of shared pretraining exposure and train–validation differences unresolved. The initialization run probes source differences expressed by one randomly initialized model with its tokenizer. Historical Table 6 used different inputs and is excluded from these comparisons. The sixth designated 1.4B derivative, nnheui/pythia-1.4b-sft-full, lacked an accessible archived checkpoint. The Lamini Docs checkpoint used the trained parent’s tokenizer.

## Rebuttal paragraph

The level-α guarantee requires the held-out sampling assumptions to hold under the null. A public train–validation split alone provides no such guarantee. On MIMIR GitHub, we fitted each model’s rules on 500 examples per source and evaluated on 200 per source. At initialization, Pythia-1.4B gave advantages of 0.060 (MIN-K20) and 0.045 (mean log probability), both below the 0.1358 reference cutoff. Independently trained GPT-Neo-1.3B, which also trained on the Pile, crossed it with both scores (0.305 and 0.355), as did the trained Pythia parent and four of five tested derivatives. Thus a model outside the Pythia lineage can exhibit a substantial shard signal. On ImageNet, the random ViT and independently trained DeiT-Base gave 0.0062 and 0.0083, both below 0.0438. These controls measure detectability on the tested models. False-positive calibration under the observed shifts remains open.

[Full numerical results](results.csv) · [Run methods and revisions](../../../../artifacts/pile_controls_reconstruction/20261124T160154122852Z/METHODS.md) · [Submitted ImageNet reference](../../../vit_provenance/results/submitted_table_8.csv) · [DeiT reference](../../../vit_provenance/results/2026-11-23-hard-negative/observed_summary.json)

All 16 thresholds were independently refitted from the saved construction scores. The resulting held-out decisions agreed with the reported values for both scores on all eight checkpoints.
