import json
import subprocess

path = "experiment"

hashes = []
if len(hashes) == 0:
    with open(f"{path}/generated_architectures.json", "r") as f:
        archis = json.load(f)
    for model_hash, model in archis.items():
        hashes.append(model_hash)


step = 0
run = False

for i, hash_value in enumerate(hashes):
    if hash_value == "4250ed331a7d7208383962dccbb184f9d12439bb44a2546cf92e0aed58094502":
        run = True
    # if i < 20 * step:
    #     continue
    # if i >= 20 * (step + 1):
    #     break
    if run:
        sbatch_cmd = f"sbatch --export=model_hash={hash_value},path={path} --job-name=zc-{hash_value} --output=output/zc-{hash_value}.out zzz_slurm/job.slurm"
        subprocess.call(sbatch_cmd.split())
