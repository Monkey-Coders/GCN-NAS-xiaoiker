import json
import subprocess
import os


path = f"experiment"
with open(f"{path}/generated_architectures.json", "r") as f:
    archis = json.load(f)

hashes = []

for model_hash, model in archis.items():
    if len(model["weights"]) == 4 or len(model["weights"]) == 6:
        constraint = "A100|V100"
    else:
        constraint = "A100"
    
    # find model weights in {path}/run/{models_hash} and get the weights .pt file with the highest epoch
    # if no weights are found, set weights to None and max_epoch to None
    # if weights are found, set weights to the weights .pt file and max_epoch to the epoch of the weights .pt file
    path_to_weights = f"{path}/run/{model_hash}"
    files = os.listdir(path_to_weights)
    weights = [file for file in files if file.endswith(".pt")]
    if len(weights) == 0:
        hashes.append(
            {
                "model_hash": model_hash,
                "weights": None,
                "max_epoch": None,
                "constraint": constraint
            }
        )
        continue
    
    max_epoch = 0
    max_weights = ""
    for weight in weights:
        epoch = int(weight.split("-")[1])
        if epoch > max_epoch:
            max_epoch = epoch
            max_weights = weight


    hashes.append(
        {
            "model_hash": model_hash,
            "weights": f"{path_to_weights}/{max_weights}",
            "max_epoch": max_epoch,
            "constraint": constraint
        }
    )

for hash_value in hashes:
    # Define the command to run the sbatch job with the hash value
    sbatch_cmd = f"sbatch --export=model_hash={hash_value['model_hash']},weights={hash_value['weights']},start_epoch={hash_value['max_epoch']},path={path} --job-name={hash_value['model_hash']} --output=output/{hash_value['model_hash']}.out --constraint={hash_value['constraint']} zzz_slurm/job.slurm"
    # print(sbatch_cmd)
    # Submit the sbatch job using subprocess
    subprocess.call(sbatch_cmd.split())
