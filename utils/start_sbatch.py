import json
import subprocess



for i in [8, 10]:
    path = f"architectures_{i}"
    with open(f"{path}/generated_architectures.json", "r") as f:
        archis = json.load(f)

    hashes = []

    for model_hash, model in archis.items():
        hashes.append(
            {
                "model_hash": model_hash,
                "weights": None,
                "max_epoch": None
            }
        )
    if i == 4 or i == 6:
        constraint = "A100|V100"
    else:
        constraint = "A100"
        
    for hash_value in hashes:
        # Define the command to run the sbatch job with the hash value
        sbatch_cmd = f"sbatch --export=model_hash={hash_value['model_hash']},weights={hash_value['weights']},start_epoch={hash_value['max_epoch']},path={path} --job-name=a{i}-{hash_value['model_hash']} --output=output/a{i}-{hash_value['model_hash']}.out --constraint={constraint} zzz_slurm/job.slurm"
        # print(sbatch_cmd)
        # Submit the sbatch job using subprocess
        subprocess.call(sbatch_cmd.split())
