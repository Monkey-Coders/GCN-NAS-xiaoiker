import json
import math
import os
import subprocess


path = "experiment"

Proxies = [
"epe_nas",
"fisher",
"flops",
"grad_norm",
"grad_sign",
"grasp",
"jacov",
"l2_norm",
"nwot",
"params",
"plain",
"snip",
"synflow",
"zen",]

def contains_nan(dictionary):
    for value in dictionary.values():
        if isinstance(value, float) and math.isnan(value):
            return True
        elif isinstance(value, dict):
            if contains_nan(value):
                return True
    return False

hashes = []
zc = ["zero_cost_scores"]
for i in range(0, 10):
    zc.append(f"zero_cost_scores_{i}")
for i in range(11, 46, 2):
    zc.append(f"zero_cost_scores_{i}")
if len(hashes) == 0:
    data = os.listdir(f"{path}/data")
    for file in data:
        with open(f"{path}/data/{file}", "r") as f:
            model = json.load(f)
            model_hash = file.split(".")[0]
            for zc_key in zc:
                if zc_key not in model[model_hash]:
                    hashes.append(model_hash)
                    break
                else:
                    should_break = False
                    for proxy in Proxies:
                        if proxy not in model[model_hash][zc_key]:
                            hashes.append(model_hash)
                            should_break = True
                            break
                    if should_break:
                        break
            if contains_nan(model[model_hash]):
                hashes.append(model_hash)
            # hashes.append(model_hash)

hashes = list(set(hashes))
print(len(hashes))
# for i, hash_value in enumerate(hashes):
#     account = "share-ie-idi"
# #     if i % 2 == 0:
# #         account = "ie-idi"
        
#     sbatch_cmd = f"sbatch --export=model_hash={hash_value},path={path} --job-name=zc-{hash_value} --output=output/zc-{hash_value}.out --account={account} zzz_slurm/job.slurm"
#     subprocess.call(sbatch_cmd.split())
