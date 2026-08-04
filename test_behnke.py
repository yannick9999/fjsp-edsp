"""Seeded EDSP evaluation on the Behnke benchmark set only, for one seed.

Run this before the full test.sh suite as a quick sanity check that the
trained policies work at all, before spending time on all configured
test sets.

Loads the final policy set of one seed (save/{exp}/seed{S}/), runs every
policy greedily on every Behnke instance, and reports the best makespan
(min over policies, the EDSP inference) and a parallel runtime proxy (max
over policies, since the policies run in parallel).

Outputs, per seed:
  results/seed{S}/results_behnke.json
  results/seed{S}/results_behnke_seed{S}.xlsx   two sheets: makespan, run_time

Behnke folder is fixed to data/benchmarks/banke, independent of the
test.datasets list in config.json.
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

BEHNKE_FOLDER = os.path.join("data", "benchmarks", "banke")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    seed = args.seed
    exp = cfg["experiment"]["name"]

    if not os.path.isdir(BEHNKE_FOLDER):
        raise RuntimeError(f"Behnke folder not found: {BEHNKE_FOLDER}")

    model_dir = os.path.join(cfg["paths"]["save_root"], exp, f"seed{seed}")
    with open(os.path.join(model_dir, "model_params.json"), "r") as f:
        model_params = json.load(f)
    if len(model_params) == 0:
        raise RuntimeError(f"No policies in {model_dir}/model_params.json")

    results_dir = os.path.join(cfg["paths"]["results_root"], f"seed{seed}")
    os.makedirs(results_dir, exist_ok=True)

    files = sorted(x for x in os.listdir(BEHNKE_FOLDER)
                   if os.path.isfile(os.path.join(BEHNKE_FOLDER, x)) and x.endswith(".fjs"))
    if len(files) == 0:
        raise RuntimeError(f"No .fjs instances found in {BEHNKE_FOLDER}")

    print(f"[test_behnke seed={seed}] device={('cuda' if torch.cuda.is_available() else 'cpu')} "
          f"policies={len(model_params)} instances={len(files)}", flush=True)

    # metadata (graph schema) is constant across instances, take it from the first one
    boot_env = FJSSPEnv([load_instance(BEHNKE_FOLDER, files[0])])
    metadata = boot_env.reset().metadata()

    agents = build_agents(model_params, model_dir, metadata)

    makespan_rows = []
    runtime_rows = []
    set_results = []

    with torch.no_grad():
        for f in files:
            instance = load_instance(BEHNKE_FOLDER, f)
            per_policy_mk = []
            per_policy_time = []
            for agent, mask_option, sel_k in agents:
                t0 = time.time()
                mk = greedy_rollout(agent, mask_option, sel_k, instance)
                per_policy_time.append(time.time() - t0)
                per_policy_mk.append(mk)

            best_mk = min(per_policy_mk)
            parallel_time = max(per_policy_time)

            makespan_rows.append({"dataset": "behnke", "instance": f, "makespan": best_mk})
            runtime_rows.append({"dataset": "behnke", "instance": f, "run_time": parallel_time})
            set_results.append({"name": f, "makespan": best_mk, "time": parallel_time})
            print(f"[test_behnke seed={seed}] behnke/{f} makespan={best_mk} time={parallel_time:.3f}s", flush=True)

    with open(os.path.join(results_dir, "results_behnke.json"), "w") as out:
        json.dump({"results": set_results}, out)

    xlsx_path = os.path.join(results_dir, f"results_behnke_seed{seed}.xlsx")
    with pd.ExcelWriter(xlsx_path) as writer:
        pd.DataFrame(makespan_rows).to_excel(writer, sheet_name="makespan", index=False)
        pd.DataFrame(runtime_rows).to_excel(writer, sheet_name="run_time", index=False)

    print(f"[test_behnke seed={seed}] done, wrote {xlsx_path}", flush=True)


if __name__ == "__main__":
    main()
