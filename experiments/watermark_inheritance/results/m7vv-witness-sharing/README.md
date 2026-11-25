# Witness sharing and causal ancestry

The m7VV Q4 response studies three Pythia-160M histories on the same prompts,
watermark key, and Dolly rows. A descends from the designated parent trained
on \(S\). B learns from the same realized \(S\) through a separate training run.
C learns from an independent same-key completion shard \(S''\). All three
take nine witness-training epochs and one Dolly epoch. The audit compares
\(S\) with \(S'\) on 1,600 held-out prompt pairs after calibrating on 400.

| History | Accepted \(S\) | Accepted \(S'\) | Signed advantage | Decision at 0.0679 |
| --- | ---: | ---: | ---: | --- |
| A · Descendant | 820 | 601 | 0.1369 | Reject |
| B · Independent training on \(S\) | 998 | 843 | 0.0969 | Reject |
| C · Independent training on \(S''\) | 277 | 271 | 0.0038 | Fail to reject |

B shows how direct access to the witness can produce a positive audit for an
independent model. [Audit table](audit.csv) · [Checkpoint lineage digests](lineages.json)

## Verify the saved results

```bash
python -m experiments.watermark_inheritance.runners.verify_m7vv
```

The saved score CSV contains paired numeric scores and split labels. The
verifier fits thresholds on the construction scores and recomputes the held-out
advantages, exact paired-swap p-values, and decisions.

## Reproduce from upstream inputs

The [protocol](../../configs/m7vv_witness_sharing.json) pins the model,
tokenizer, dataset and upstream watermark code, all training and sampling
seeds, and the expected hashes of the source and prepared files. The
Pythia-160M training uses BF16 autocast with FP32 master weights, effective
batch size 32, and three three-epoch learning-rate cycles at peak learning rate 5e-5. The Dolly stage uses one epoch at peak \(10^{-5}\) and shares the
same 512 source rows across all arms.

```bash
run=artifacts/m7vv_witness_sharing
inputs="$run/inputs"
mkdir -p "$inputs/sources"
hf download EleutherAI/pythia-160m \
  --revision 50f5173d932e8e61f858120bcb800b97af589f46 \
  --include '*.safetensors' '*.json' --local-dir "$inputs/base"
hf download Salesforce/wikitext --repo-type dataset \
  --revision b08601e04326c79dfdd32d625aee71d232d685c3 \
  --include 'wikitext-103-raw-v1/train-*-of-00002.parquet' \
  --local-dir "$inputs/wiki"
cp "$inputs"/wiki/wikitext-103-raw-v1/train-*-of-00002.parquet "$inputs/sources/"
hf download databricks/databricks-dolly-15k databricks-dolly-15k.jsonl --repo-type dataset \
  --revision bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a --local-dir "$inputs/sources"
git clone https://github.com/facebookresearch/radioactive-watermark "$inputs/sources/radioactive-watermark"
git -C "$inputs/sources/radioactive-watermark" checkout --detach 19c6dc90e7a0dddb1a241b73e7fae708b68f024e
python -m experiments.watermark_inheritance.runners.m7vv_confirm_shards prepare --run-dir "$run"
python -m experiments.watermark_inheritance.runners.m7vv_confirm_shards generate --run-dir "$run"
python -m experiments.watermark_inheritance.runners.m7vv_confirm_targets --run-dir "$run"
```

Generation and training require a BF16-capable CUDA GPU. The preparation step
reconstructs the exact prompt exclusions from the frozen seeds and verifies the
published prompt and Dolly hashes. The original run used an A100-SXM4-40GB.
