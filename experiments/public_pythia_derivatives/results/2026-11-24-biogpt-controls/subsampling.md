# BioGPT-Large and random Pythia on the held-out MIMIR GitHub split

## Rejection frequencies

For each model and score, we froze the threshold fitted on the 500 member and 500 nonmember construction examples. The held-out pool contains 200 distinct texts per source. For one replicate, we independently shuffled each source and took its first *m* items without replacement. The same rows were used for both models and both scores. Prefixes of each shuffle supplied the five sample sizes. Across 10,000 replicates we used NumPy's default generator with seed 3324. A rejection required the absolute difference in acceptance rates to exceed sqrt(log(2/α)/m).

Each cell gives the number of rejections in 10,000 draws, followed by the exact finite-pool probability as a percentage. For example, a count of 50 means a simulated rejection frequency of 0.5%. At *m* = 200 only one pair of subsets exists, so every draw has the same outcome.

| α | *m* per source | BioGPT MIN-K20 | Random Pythia MIN-K20 | BioGPT mean log probability | Random Pythia mean log probability |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 25 | 0 (0%) | 0 (0.003059%) | 0 (4.209e-07%) | 7 (0.07456%) |
| 0.01 | 50 | 0 (0%) | 0 (0.001665%) | 0 (4.647e-10%) | 6 (0.04524%) |
| 0.01 | 100 | 0 (0%) | 1 (0.0001308%) | 0 (0%) | 0 (0.006722%) |
| 0.01 | 150 | 0 (0%) | 0 (4.109e-08%) | 0 (0%) | 0 (1.955e-05%) |
| 0.01 | 200 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| 0.025 | 25 | 0 (0%) | 2 (0.01569%) | 0 (8.001e-06%) | 26 (0.2174%) |
| 0.025 | 50 | 0 (0%) | 3 (0.02117%) | 0 (2.498e-07%) | 27 (0.2231%) |
| 0.025 | 100 | 0 (0%) | 1 (0.005469%) | 0 (0%) | 4 (0.06584%) |
| 0.025 | 150 | 0 (0%) | 0 (1.335e-05%) | 0 (0%) | 0 (0.0006459%) |
| 0.025 | 200 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| 0.05 | 25 | 0 (0%) | 7 (0.06867%) | 0 (0.0001166%) | 60 (0.5744%) |
| 0.05 | 50 | 0 (0%) | 7 (0.06529%) | 0 (3.575e-06%) | 45 (0.4581%) |
| 0.05 | 100 | 0 (0%) | 3 (0.01618%) | 0 (0%) | 13 (0.1303%) |
| 0.05 | 150 | 0 (0%) | 0 (0.0003438%) | 0 (0%) | 1 (0.005005%) |
| 0.05 | 200 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| 0.1 | 25 | 0 (0%) | 32 (0.2577%) | 0 (0.001318%) | 140 (1.381%) |
| 0.1 | 50 | 0 (0%) | 22 (0.1834%) | 0 (3.948e-05%) | 108 (0.8941%) |
| 0.1 | 100 | 0 (0%) | 14 (0.1131%) | 0 (5.216e-09%) | 47 (0.4557%) |
| 0.1 | 150 | 0 (0%) | 0 (0.005682%) | 0 (0%) | 3 (0.03114%) |
| 0.1 | 200 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |

For the exact probabilities we summed the product of the two hypergeometric probability masses over every pair of acceptance counts that crossed the cutoff. This conditions on the two observed pools, each model and its fitted threshold. The balance between member and nonmember acceptance, along with the sampling fraction, determines how often that fixed audit crosses. The experiment measures sensitivity to held-out row selection. Estimating a population false-positive rate across new public splits would require a sampling design for those splits.

The accompanying CSV includes the cutoff, pool acceptance counts, signed advantage, raw rejection counts and both probabilities for every cell. [Subsampling CSV](subsampling.csv)
