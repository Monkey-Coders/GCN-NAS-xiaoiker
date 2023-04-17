import json
import math
import subprocess

import numpy as np

path = "experiment"

hashes = []
if len(hashes) == 0:
    with open(f"{path}/generated_architectures.json", "r") as f:
        archis = json.load(f)
    for model_hash, model in archis.items():
        if "zero_cost_scores" not in model:
            hashes.append(model_hash)
            continue
        if any(math.isnan(val["time"]) for val in model["zero_cost_scores"].values()):
            hashes.append(model_hash)
            continue
        
        for i in range(0, 10):
            if f"zero_cost_scores_{i}" not in model:
                hashes.append(model_hash)
                break
            if any(math.isnan(val["time"]) for val in model[f"zero_cost_scores_{i}"].values()):
                hashes.append(model_hash)
                break
for hash_value in hashes:
    sbatch_cmd = f"sbatch --export=model_hash={hash_value} --job-name=zc-{hash_value} --output=output/zc-{hash_value}.out zzz_slurm/zc.slurm"
    subprocess.call(sbatch_cmd.split())
