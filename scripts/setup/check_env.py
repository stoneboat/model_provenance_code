"""Environment checker for model_provenance Phase 1.

Prints a concise report of installed packages, hardware, and required
directory existence.  Exits with code 1 if any essential package is missing.
"""

import os
import sys
import zlib

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

REQUIRED_DIRS = [
    "data/raw",
    "data/processed",
    "data/scores",
    "outputs/runs",
    "outputs/reports",
    "outputs/figures",
]


def _try_import(module_name: str):
    try:
        mod = __import__(module_name)
        version = getattr(mod, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, None


def main() -> int:
    print("=" * 60)
    print("  model_provenance — environment check")
    print("=" * 60)

    print(f"\n[python]")
    print(f"  version    : {sys.version}")
    print(f"  executable : {sys.executable}")
    print(f"  cwd        : {os.getcwd()}")
    print(f"  repo root  : {REPO_ROOT}")

    # ------------------------------------------------------------------ #
    # Package checks
    # ------------------------------------------------------------------ #
    essential_missing = []
    checks = [
        ("torch",          True,  "torch"),
        ("transformers",   True,  "transformers"),
        ("datasets",       True,  "datasets"),
        ("sklearn",        True,  "scikit-learn"),
        ("numpy",          True,  "numpy"),
        ("matplotlib",     False, "matplotlib"),
        ("pandas",         False, "pandas"),
        ("tqdm",           False, "tqdm"),
        ("yaml",           False, "pyyaml"),
    ]

    print("\n[packages]")
    for import_name, essential, display_name in checks:
        ok, version = _try_import(import_name)
        tag = "OK" if ok else ("MISSING [essential]" if essential else "MISSING [optional]")
        ver_str = version if ok else "—"
        print(f"  {display_name:<20} {tag:<25} {ver_str}")
        if not ok and essential:
            essential_missing.append(display_name)

    # zlib is stdlib; check separately
    print(f"  {'zlib':<20} {'OK':<25} {zlib.ZLIB_VERSION}")

    # ------------------------------------------------------------------ #
    # Torch / GPU
    # ------------------------------------------------------------------ #
    print("\n[torch / gpu]")
    torch_ok, _ = _try_import("torch")
    if torch_ok:
        import torch  # noqa: PLC0415
        cuda_avail = torch.cuda.is_available()
        print(f"  cuda available : {cuda_avail}")
        if cuda_avail:
            gpu_count = torch.cuda.device_count()
            print(f"  gpu count      : {gpu_count}")
            for i in range(gpu_count):
                print(f"  gpu[{i}]         : {torch.cuda.get_device_name(i)}")
        else:
            print("  gpu count      : 0  (CUDA not available — CPU only)")
    else:
        print("  torch not installed; skipping GPU check")

    # ------------------------------------------------------------------ #
    # HuggingFace env vars
    # ------------------------------------------------------------------ #
    print("\n[hf environment]")
    for var in ("HF_HOME", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE"):
        print(f"  {var:<25} {os.environ.get(var, '[not set]')}")

    # ------------------------------------------------------------------ #
    # Required directories
    # ------------------------------------------------------------------ #
    print("\n[required directories]")
    all_dirs_ok = True
    for rel_path in REQUIRED_DIRS:
        full = os.path.join(REPO_ROOT, rel_path)
        exists = os.path.isdir(full)
        status = "OK" if exists else "MISSING"
        print(f"  {rel_path:<30} {status}")
        if not exists:
            all_dirs_ok = False

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    if essential_missing:
        print(f"[FAIL] Missing essential packages: {', '.join(essential_missing)}")
        print("       Run: pip install -r requirements.txt")
        print("=" * 60)
        return 1
    if not all_dirs_ok:
        print("[WARN] Some required directories are missing.")
        print("       Run: bash scripts/setup/setup_local_env.sh")
    else:
        print("[OK]   Environment looks good for Phase 1 scaffold.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
