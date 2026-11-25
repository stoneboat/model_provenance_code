# Matched neural development histories for m7VV Q4

The named candidate parent P is the stage-one output in arm A. All three branches start from the same immutable pretrained Pythia-160M weights, use the same optimizer protocol, and receive the same fixed Dolly rows. Separate training random seeds give each branch its own development history.

| Target | Stage one | Stage two | Descendant of P? | Does exclusion hold for P? |
|---|---|---|---|---|
| A | Base trains on all of S, producing P | Continue from P on Dolly | Yes | Yes. Realized witness randomness reaches A through P. |
| B | Fresh base trains on exactly the same S, producing its own checkpoint | Continue from that checkpoint on Dolly | No | No. Direct access to S bypasses P. |
| C | Fresh base trains on independent S″, producing its own checkpoint | Continue from that checkpoint on Dolly | No | Yes, vacuously. Given the shared background, there is no path from the realized S draw to C. |

The conditioning background includes the frozen generator, tokenizer, prompts, watermark key, sampling law, pretrained weights, and downstream corpus. Parameters and matching prompts are shared across S, S′, and S″. Sampling streams are separate. The Dolly subset is selected from the original Dolly data by a fixed independent seed with no filtering against any generated text. S′ appears only on the auditing side. Training on all 2,000 S examples in A and B is intentional: the 400/1,600 split holds examples out from constructing the detector.

## Decisions fixed before evaluation

- Primary readout is the original keyed Maryland watermark score. For each target, fit the checkpoint-specific threshold on the 400 construction pairs, predict witness for scores strictly above it, maximize balanced accuracy, and choose the largest threshold in a tie.
- On 1,600 untouched evaluation pairs compute the signed gap in acceptance rates, S minus S′, and its absolute value. At nominal per-target alpha 0.05 reject when the absolute gap exceeds sqrt(2 log(2/0.05)/1600) = 0.0679051. Also give the two-sided exact within-pair swap p-value. The cutoff and swap p-value are distinct; the former determines the reported decision. Round only for display.
- MIN-K20, when reported, is a secondary diagnostic. The experiment has one realized key, one set of prompts and one trained target per arm. Its decisions are observations for these histories, not estimates of power or error rates over repeated model developments.
- Use a training-throughput pilot and a disjoint generation timing pilot. Choose completion length and batching for compute alone. Use three complete stage-one epochs per arm and one Dolly epoch over 512 fixed examples. The effective batch size is 32. Retain the final three targets and the information necessary to reproduce their ancestry.

## Timing decision and reconstruction

Four pilot optimizer updates took 1.20 seconds at 320 tokens per training example. The generation pilot drew 16 triples, from 16 articles outside the main 2,000, in 5.55 seconds at a cap of 64 new tokens. A linear projection of 46 minutes for the main generation, plus under eight minutes of estimated training, left room for inference and scoring. We therefore fixed the original 256-token cap for every main arm before training or evaluating any experimental target. The generator stops at EOS. There is one main prompt per distinct article. All settings here refer to this experiment.

Use the repository root as working directory. The local Python environment for this run is `.cache/vit-venv/bin/python`. Input files and scores live under `artifacts/m7vv_q4`. The base repository commit before adding these runners was `f2c399ca034b49d35d950552b0a5c148b699154c`.

| Input | Revision | Location |
|---|---|---|
| EleutherAI/pythia-160m, final checkpoint and tokenizer | `50f5173d932e8e61f858120bcb800b97af589f46` | `artifacts/m7vv_q4/base/` |
| Salesforce/wikitext, wikitext-103-raw-v1, train | `b08601e04326c79dfdd32d625aee71d232d685c3` | Both training parquet parts in `artifacts/m7vv_q4/sources/` |
| facebookresearch/radioactive-watermark | `19c6dc90e7a0dddb1a241b73e7fae708b68f024e` | `artifacts/m7vv_q4/sources/radioactive-watermark/` |
| databricks/databricks-dolly-15k, train | `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` | Cached `artifacts/dolly_matched/data/databricks-dolly-15k.jsonl` and `source.json` |

For an independent rerun, use a fresh output tree and the same pinned inputs. The training preparation takes only the base and original Dolly data as inputs. The repository's sampler handles prompt selection, Maryland greenlists, temperature 0.8 and top-p 0.95. Generation uses key 101, context of two tokens, green fraction 0.25 and logit bias 3.0. The three branches receive distinct initialization-of-training RNG seeds; their starting pretrained weights are identical.

```bash
py=.cache/vit-venv/bin/python
run=artifacts/m7vv_q4
wm=experiments.watermark_inheritance
"$py" -m "$wm.runners.matched_q4_train" --root "$run" prepare
"$py" -m "$wm.m7vv_shards" prepare --run "$run"
"$py" -m "$wm.m7vv_shards" pilot --run "$run"
"$py" -m "$wm.m7vv_shards" full --run "$run"
for target in A B C; do
    "$py" -m "$wm.runners.matched_q4_train" --root "$run" "$target"
done
for target in A B C; do
    "$py" -m "$wm.score" --run-dir "$run" --checkpoint-name "$target" \
        --checkpoint-path "$run/train/$target/checkpoints/target_epoch1"
done
"$py" -m "$wm.matched_process_audit" --run-dir "$run" --output-dir "$run/audit"
```
