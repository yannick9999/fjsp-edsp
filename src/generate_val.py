import json
import os
import random
import numpy as np

from src.generator import generate_instance_list
from src.parsedata import get_data, parse
from src.solver import solve_fjsp


def _atomic_dump(obj, path):
    """Write JSON atomically so a killed job never leaves a half written file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def generate_val(cfg, seed, paths):
    """Generate a per seed validation set and solve it with CP-SAT.

    This module must stay free of any torch import. CP-SAT and torch in the
    same process have caused 'double free or corruption' crashes before, so
    validation generation runs in its own torch free process.
    """
    # deterministic per seed instance generation (generator uses the random module)
    random.seed(seed)
    np.random.seed(seed)

    v = cfg["validation"]
    val_config = {
        "n_cases": v["n_cases"],
        "range_jobs": tuple(v["range_jobs"]),
        "range_machines": tuple(v["range_machines"]),
        "range_op_per_job": tuple(v["range_op_per_job"]),
        "max_processing": v["max_processing"],
    }
    time_limit = v["cp_time_limit"]

    val_dir = os.path.dirname(paths["val_set"])
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(paths["candidate_dir"], exist_ok=True)

    list_instances = generate_instance_list(**val_config)

    list_scores = []
    for k, instance in enumerate(list_instances):
        jobs, operations, info, maximum = get_data(parse(instance))
        _, score, _ = solve_fjsp(jobs, operations, info, time_limit=time_limit)
        list_scores.append({
            "score": str(int(score)),
            "jobs": jobs,
            "operations": operations,
            "maximum": maximum,
            "num_machines": info["machinesNb"],
        })
        print(f"[val seed={seed}] solved {k + 1}/{len(list_instances)} score={int(score)}", flush=True)

    _atomic_dump(list_scores, paths["val_set"])

    # running best gap per instance, used during BO for candidate selection
    val_results = [{"best": 0.4, "name": None} for _ in range(len(list_scores))]
    _atomic_dump(val_results, paths["val_results"])

    # empty candidate registry for this seed
    _atomic_dump([], paths["candidate_params"])

    print(f"[val seed={seed}] done, {len(list_scores)} instances written to {paths['val_set']}", flush=True)
