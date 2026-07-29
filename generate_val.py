"""Standalone, torch free validation set generation for one seed.

Run this before training (see slurm/generate_val.sh). It produces:
  val/seed{S}/validation_set.json       instances + CP-SAT makespan
  val/seed{S}/validation_results.json   running best gap per instance
  candidate_models/seed{S}/model_params.json   empty candidate registry

Do NOT import torch anywhere in this file or its import chain.
"""
import argparse
import json
import os

from src.generate_val import generate_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    seed = args.seed
    val_dir = os.path.join("val", f"seed{seed}")
    cand_dir = os.path.join("candidate_models", f"seed{seed}")
    paths = {
        "val_set": os.path.join(val_dir, "validation_set.json"),
        "val_results": os.path.join(val_dir, "validation_results.json"),
        "candidate_dir": cand_dir,
        "candidate_params": os.path.join(cand_dir, "model_params.json"),
    }

    if os.path.exists(paths["val_set"]):
        print(f"[val seed={seed}] {paths['val_set']} already exists, skipping generation")
        return

    generate_val(cfg, seed, paths)


if __name__ == "__main__":
    main()
