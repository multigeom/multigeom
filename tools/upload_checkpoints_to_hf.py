"""
author: laquitainesteeve@gmail.com

Upload the trained checkpoints in runs/ to a Hugging Face model repository,
keeping the repository's own layout so they can be restored with one call:

    encoders/<encoder>/<area>/best_model.pt
    diffusion/<encoder>/<area>/final_model.pt

Authenticate first, with either of

    huggingface-cli login            # interactive, stores a token
    export HF_TOKEN=hf_xxx           # a token with "write" access

Usage:
    python tools/upload_checkpoints_to_hf.py --repo-id steevelaquitaine/multigeom --dry-run
    python tools/upload_checkpoints_to_hf.py --repo-id steevelaquitaine/multigeom  # private repo
    python tools/upload_checkpoints_to_hf.py --repo-id steevelaquitaine/multigeom --public

Example on my mac (where I have the checkpoints):

    python upload_checkpoints_to_hf.py --runs-dir /Users/steeve_laquitaine/Desktop/info-manifold-paper/package/MIG_checkpoints_for_Steeve/cfg_models/runs --repo-id steevelaquitaine/multigeom --dry-run
    python upload_checkpoints_to_hf.py --runs-dir /Users/steeve_laquitaine/Desktop/info-manifold-paper/package/MIG_checkpoints_for_Steeve/cfg_models/runs --repo-id steevelaquitaine/multigeom --public # (20 secs)

The repository is created private unless --public is given. Restore with

    huggingface-cli download steevelaquitaine/multigeom --local-dir runs
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent

PATTERNS = ["encoders/*/*/best_model.pt", "diffusion/*/*/final_model.pt"]

CARD = """---
license: mit
tags:
  - neuroscience
  - diffusion-models
  - fisher-information
---

# Multiscale information geometry: trained checkpoints

Encoder and conditional-diffusion checkpoints for the NeurIPS 2026 paper. Code:
<CODE_URL>.

```
encoders/<encoder>/<area>/best_model.pt      # stage 1: frozen backbone + Gaussian readout
diffusion/<encoder>/<area>/final_model.pt    # stage 3: conditional diffusion model
```

`<encoder>` is `inception`, `resnet50` or `onoff`; `<area>` is `V1` or `V4`.

Restore into the code repository with `huggingface-cli download {repo_id} --local-dir runs`.
"""


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, help="e.g. <user-or-org>/multigeom")
    p.add_argument("--runs-dir", default=str(ROOT / "runs"))
    p.add_argument("--public", action="store_true", help="create the repo public (default: private)")
    p.add_argument("--dry-run", action="store_true", help="list the files and sizes, upload nothing")
    p.add_argument("--code-url", default="https://github.com/<owner>/MultiscaleInformationGeometry")
    args = p.parse_args(argv)

    runs = Path(args.runs_dir)
    files = sorted(f for pat in PATTERNS for f in runs.glob(pat))
    if not files:
        sys.exit(f"No checkpoints found under {runs} (expected {PATTERNS}).")

    total = 0
    for f in files:
        size = f.stat().st_size
        total += size
        print(f"{size / 1e6:9.1f} MB  {f.relative_to(runs)}")
    print(f"{len(files)} files, {total / 1e9:.2f} GB total")
    if args.dry_run:
        return

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id, repo_type="model", folder_path=str(runs),
        allow_patterns=PATTERNS, commit_message="Upload trained checkpoints",
    )
    with tempfile.TemporaryDirectory() as tmp:
        readme = Path(tmp) / "README.md"
        readme.write_text(CARD.replace("<CODE_URL>", args.code_url).replace("{repo_id}", args.repo_id))
        api.upload_file(path_or_fileobj=str(readme), path_in_repo="README.md",
                        repo_id=args.repo_id, repo_type="model", commit_message="Add model card")
    vis = "public" if args.public else "private"
    print(f"Uploaded to https://huggingface.co/{args.repo_id} ({vis})")


if __name__ == "__main__":
    main()
