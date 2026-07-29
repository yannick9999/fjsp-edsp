"""Seeded EDSP training for one seed.

Runs the Bayesian optimization loop from the paper (Optuna, TPE sampler) to
generate candidate policies, then clusters them into the final policy set.
Everything is namespaced by seed so three array tasks (seed 0, 1, 2) never
touch each other's files.

Prerequisite: the per seed validation set must already exist under
val/seed{S}/ (created by generate_val.py, which is torch free). This split
keeps CP-SAT out of the torch process.

Final output: save/{experiment.name}/seed{S}/  (the 6 policies + params).
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from src.train import train as train_policy
from src.cluster_policies import cluster_policies


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    seed = args.seed
    set_seed(seed)

    exp = cfg["experiment"]["name"]
    save_root = cfg["paths"]["save_root"]

    val_dir = os.path.join("val", f"seed{seed}")
    cand_dir = os.path.join("candidate_models", f"seed{seed}")
    save_dir = os.path.join(save_root, exp, f"seed{seed}")
    os.makedirs(cand_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    paths = {
        "val_set": os.path.join(val_dir, "validation_set.json"),
        "val_results": os.path.join(val_dir, "validation_results.json"),
        "candidate_dir": cand_dir,
        "candidate_params": os.path.join(cand_dir, "model_params.json"),
    }

    if not os.path.exists(paths["val_set"]):
        raise FileNotFoundError(
            f"Missing {paths['val_set']}. Run generate_val.py --seed {seed} first."
        )

    ss = cfg["search_space"]
    fixed_heads = ss["heads"]
    n_instances = cfg["train"]["n_instances"]
    lr_lo, lr_hi = ss["lr"][0], ss["lr"][1]

    def objective(trial):
        # per trial deterministic seed, so a resumed study reproduces each trial
        set_seed(seed * 100000 + trial.number)
        return train_policy(
            trial.suggest_int("max_episodes", ss["max_episodes"][0], ss["max_episodes"][1]),
            trial.suggest_int("train_freq", ss["train_freq"][0], ss["train_freq"][1]),
            trial.suggest_int("new_freq", ss["new_freq"][0], ss["new_freq"][1]),
            n_instances,
            trial.suggest_categorical("mask_option", ss["mask_option"]),
            trial.suggest_int("sel_k", ss["sel_k"][0], ss["sel_k"][1]),
            trial.suggest_categorical("batch_size", ss["batch_size"]),
            trial.suggest_float("lr", lr_lo, lr_hi, log=True),
            trial.suggest_categorical("hidden_channels", ss["hidden_channels"]),
            trial.suggest_int("num_layers", ss["num_layers"][0], ss["num_layers"][1]),
            fixed_heads,
            trial.suggest_int("j_max", ss["j_max"][0], ss["j_max"][1]),
            trial.suggest_int("j_min", ss["j_min"][0], ss["j_min"][1]),
            trial.suggest_int("m_max", ss["m_max"][0], ss["m_max"][1]),
            trial.suggest_int("m_min", ss["m_min"][0], ss["m_min"][1]),
            trial.suggest_int("op_max", ss["op_max"][0], ss["op_max"][1]),
            trial.suggest_int("max_processing", ss["max_processing"][0], ss["max_processing"][1]),
            paths,
        )

    # persistent per seed study, so a preempted job resumes instead of restarting
    storage = f"sqlite:///{os.path.join(save_dir, 'optuna.db')}"
    study = optuna.create_study(
        storage=storage,
        study_name=f"{exp}_seed{seed}",
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=seed),
    )

    target = cfg["bo"]["n_trials"]
    n_complete = len([t for t in study.trials if t.state == TrialState.COMPLETE])
    remaining = max(0, target - n_complete)
    print(f"[train seed={seed}] {n_complete}/{target} trials complete, running {remaining} more", flush=True)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)

    cluster_policies(
        cfg["final_policies"],
        cand_dir,
        paths["candidate_params"],
        save_dir,
    )
    print(f"[train seed={seed}] done, final policies in {save_dir}", flush=True)


if __name__ == "__main__":
    main()
