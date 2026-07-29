"""Seeded EDSP evaluation over all test sets for one seed.

Loads the final policy set of one seed (save/{exp}/seed{S}/), runs every
policy greedily on every instance of every configured test set, and reports
the best makespan (min over policies, the EDSP inference) and a parallel
runtime proxy (max over policies, since the policies run in parallel).

Outputs, per seed:
  results/seed{S}/results_{dataset}.json
  results/seed{S}/results_seed{S}.xlsx   two sheets: makespan, run_time

Add new test sets by editing the "test.datasets" list in config.json. No code
change is needed. Instance names must end in .fjs.
"""
import argparse
import json
import os
import time

import pandas as pd
import torch

from src.env import FJSSPEnv
from src.ppo import PPO
from src.parsedata import get_data, parse


def load_instance(folder, filename):
    with open(os.path.join(folder, filename), "r") as f:
        contents = f.read()
    jobs, operations, info, maximum = get_data(parse(contents))
    return {
        "jobs": jobs,
        "operations": operations,
        "maximum": maximum,
        "num_machines": info["machinesNb"],
    }


def build_agents(model_params, model_dir, metadata):
    agents = []
    for p in model_params:
        agent = PPO(
            0.001, 0.001, 1, 3, 0.2,
            None, metadata,
            p["hidden_channels"], p["num_layers"], 64, p["heads"],
        )
        agent.load(os.path.join(model_dir, p["name"]))
        agents.append((agent, p["mask_option"], p["sel_k"]))
    return agents


def greedy_rollout(agent, mask_option, sel_k, instance):
    env = FJSSPEnv([instance], mask_option, sel_k)
    agent.env = env
    state = env.reset()
    for q in range(1, 10 ** 10):
        action = agent.select_action(state, 2, q)
        state, _, done, _ = env.step(action)
        if done:
            return env.mk


def discover_datasets(cfg):
    datasets = cfg["test"]["datasets"]
    selected = []
    for d in datasets:
        if os.path.isdir(d["path"]):
            selected.append((d["name"], d["path"]))
        else:
            print(f"[test] WARNING dataset folder not found, skipping: {d['path']}", flush=True)
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--diagnose", action="store_true",
                    help="reserved for later diagnostics, currently a no op")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    seed = args.seed
    exp = cfg["experiment"]["name"]

    model_dir = os.path.join(cfg["paths"]["save_root"], exp, f"seed{seed}")
    with open(os.path.join(model_dir, "model_params.json"), "r") as f:
        model_params = json.load(f)
    if len(model_params) == 0:
        raise RuntimeError(f"No policies in {model_dir}/model_params.json")

    results_dir = os.path.join(cfg["paths"]["results_root"], f"seed{seed}")
    os.makedirs(results_dir, exist_ok=True)

    datasets = discover_datasets(cfg)
    if len(datasets) == 0:
        raise RuntimeError("No test datasets found.")

    print(f"[test seed={seed}] device={('cuda' if torch.cuda.is_available() else 'cpu')} "
          f"policies={len(model_params)} datasets={[d[0] for d in datasets]}", flush=True)

    # metadata (graph schema) is constant across instances, take it from the first one
    first_name, first_folder = datasets[0]
    first_files = sorted(x for x in os.listdir(first_folder) if x.endswith(".fjs"))
    boot_env = FJSSPEnv([load_instance(first_folder, first_files[0])])
    metadata = boot_env.reset().metadata()

    agents = build_agents(model_params, model_dir, metadata)

    makespan_rows = []
    runtime_rows = []

    with torch.no_grad():
        for set_name, folder in datasets:
            files = sorted(x for x in os.listdir(folder)
                           if os.path.isfile(os.path.join(folder, x)) and x.endswith(".fjs"))
            set_results = []
            for f in files:
                instance = load_instance(folder, f)
                per_policy_mk = []
                per_policy_time = []
                for agent, mask_option, sel_k in agents:
                    t0 = time.time()
                    mk = greedy_rollout(agent, mask_option, sel_k, instance)
                    per_policy_time.append(time.time() - t0)
                    per_policy_mk.append(mk)

                best_mk = min(per_policy_mk)
                parallel_time = max(per_policy_time)

                makespan_rows.append({"dataset": set_name, "instance": f, "makespan": best_mk})
                runtime_rows.append({"dataset": set_name, "instance": f, "run_time": parallel_time})
                set_results.append({"name": f, "makespan": best_mk, "time": parallel_time})
                print(f"[test seed={seed}] {set_name}/{f} makespan={best_mk} time={parallel_time:.3f}s", flush=True)

            with open(os.path.join(results_dir, f"results_{set_name}.json"), "w") as out:
                json.dump({"results": set_results}, out)

    xlsx_path = os.path.join(results_dir, f"results_seed{seed}.xlsx")
    with pd.ExcelWriter(xlsx_path) as writer:
        pd.DataFrame(makespan_rows).to_excel(writer, sheet_name="makespan", index=False)
        pd.DataFrame(runtime_rows).to_excel(writer, sheet_name="run_time", index=False)

    print(f"[test seed={seed}] done, wrote {xlsx_path}", flush=True)


if __name__ == "__main__":
    main()
