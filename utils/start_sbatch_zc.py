import json
import subprocess

path = "experiment"

hashes = []
if len(hashes) == 0:
    with open(f"{path}/generated_architectures.json", "r") as f:
        archis = json.load(f)
    for model_hash, model in archis.items():
        hashes.append(model_hash)

for hash_value in hashes:
    sbatch_cmd = f"sbatch --export=model_hash={hash_value['model_hash']},path={path} --job-name=zc-{hash_value['model_hash']} --output=output/zc-{hash_value['model_hash']}.out zzz_slurm/job.slurm"
    subprocess.call(sbatch_cmd.split())
