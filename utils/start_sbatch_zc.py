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

for i, hash_value in enumerate(hashes):
    if i < 10 * step:
        continue
    if i >= 10 * (step + 1):
        break
    
    sbatch_cmd = f"sbatch --export=model_hash={hash_value},path={path} --job-name=firstRun --output=output/zc-{hash_value}.out zzz_slurm/job.slurm"
    subprocess.call(sbatch_cmd.split())
