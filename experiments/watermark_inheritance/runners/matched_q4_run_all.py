"""Wait for validated shards, train A/B/C serially, verify lineage, then release the GPU."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

from experiments.watermark_inheritance.common import load_json, write_json
from experiments.watermark_inheritance.prepare import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/m7vv_q4"))
    parser.add_argument("--deadline", required=True, help="UTC ISO timestamp for the entire experiment")
    args = parser.parse_args()
    root = args.root
    train = root / "train"
    expiry = datetime.fromisoformat(args.deadline).timestamp()

    def remaining():
        # Leave at least ten minutes for audit and packaging.
        seconds = expiry - time.time() - 600
        if seconds <= 0:
            raise TimeoutError("Training allocation expired")
        return seconds

    while not (root / "prepared/generation_ready.json").is_file():
        time.sleep(min(15, remaining()))
    states = {}
    for arm in ("A", "B", "C"):
        run = train / arm
        with (run / "train.log").open("a", buffering=1) as log:
            subprocess.run([sys.executable, "-u", "-m", "experiments.watermark_inheritance.runners.matched_q4_train",
                            "--root", str(root), arm], stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=remaining())
        states[arm] = load_json(run / "lineage.json")
    base_digest = file_sha256(root / "base/model.safetensors")
    dolly_digest = file_sha256(train / "shared/dolly.jsonl")
    for arm, state in states.items():
        witness, final = state["parent"], state["target"]
        assert witness["source_model_sha256"] == base_digest
        assert final["source_model_sha256"] == witness["output_model_sha256"]
        assert final["training_data_sha256"] == dolly_digest
        assert final["output_model_sha256"] == file_sha256(Path(final["output_model"]) / "model.safetensors")
        assert [e["examples"] for e in witness["epochs"]] == [2000]*3
        assert [e["examples"] for e in final["epochs"]] == [512]
        assert final["updates"] == 16 and witness["updates"] == 189
    assert states["A"]["parent"]["training_data_sha256"] == states["B"]["parent"]["training_data_sha256"]
    assert states["C"]["parent"]["training_data_sha256"] != states["A"]["parent"]["training_data_sha256"]
    assert len({s[stage]["output_model_sha256"] for s in states.values() for stage in ("parent","target")})==6
    assert len({s[stage]["seed"] for s in states.values() for stage in ("parent","target")})==6
    write_json(train / "final_ready.json", dict(verified_at_utc=datetime.now(timezone.utc).isoformat(),
        base_model_sha256=base_digest, dolly_training_data_sha256=dolly_digest,
        targets={name:states[name]["target"]["output_model"] for name in states},
        validation="Verified common base, exact source-to-target weights, independent seeds/weights, witness exposure, common downstream input, and full coverage."))
    print("All three training subprocesses exited. Validated targets ready for paired auditing.",flush=True)


if __name__=="__main__":
    main()
