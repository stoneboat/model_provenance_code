# Conditional-power simulation

The oT8R Q5 response compares the lower bound from Proposition 6 in §4.3
with the bound under conditionally independent held-out decisions and a
uniform conditional signal. The simulation draws independent binomial counts
for two shards with acceptance probabilities 0.55 and 0.45. It uses the
unpaired two-sided cutoff at α = 0.05 and 10,000 independent replications
per sample size. This simulation uses no models or external data.

| Samples per shard | Proposition 6 lower bound | Conditional lower bound | Empirical power |
| ---: | ---: | ---: | ---: |
| 400 | 0.44% | 0.63% | 53.86% |
| 1,600 | 5.46% | 98.67% | 99.86% |
| 8,000 | 8.03% | ≈100% | 100% |

The configuration preserves the original simulation seed and case index.
Use NumPy 1.26.4 to reproduce the reported draw counts.

```bash
python experiments/power_simulation/reproduce.py --check
python experiments/power_simulation/reproduce.py
```

[Configuration](config.json) · [Full-precision results](results.csv)
