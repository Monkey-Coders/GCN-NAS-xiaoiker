import json
import subprocess
import os

hashes = []
path = f"experiment"
with open(f"{path}/generated_architectures.json", "r") as f:
    archis = json.load(f)

hashishish = [
    # '8c6865ad978ba75c68e67d44acfc2ccfd5f0f6c073b71817e9cdc0cd3a644b93', '1655ba5af51ed6543a433d67c8b63fe91b4b003726985b20945d3cedd0209978', '4f1a471724603d9df405172472f4362466de7c40cae4c72bddee27da518036d0', '639b362701548d238d3a38af27ea08e9c6a1faa4a995a617d858eb08085ad0c9', 'f49e99b5a6386c0e3dc10791c4adef24560af45517164961dbc319337e403f85', 'fcb8f5353ce26bc587d1d02cc93d10c7a48f1f6b56552310c018a8daa09dd16e', '7cd77f5ebe61e0a79f57081019fcce8d15239e94b49037ef2e2bdd2bc8088cf9', '10c58f2d57d90c82b6f5a5b5497892836aaf94253cd672bd3e9162785462ac67', '583edd66a52dfec2fcd814245d60157841cd36944627d5ef6428cb308802f22d', 'aee9c892fa3112a139b24b772233d37ac4a7e7751c44652fb01df299e4c09ea4', 'b0cb1ae61f27a8cd317465b1980bd0651de5d2f3ab0c6fb9669e3d120f3288c9', '19cdb096f6da95d28e83b88cfa6e204b82b1e2b9af87fb7c2552509f11dba147', '9f9e271cc7edfeeeb3a4a7c33ae78bbcae26e1251529d99686e99e95e17a2507', '70c9cf32d93fa4eced8113527e14156584c1462562d249e716ee5895f3940c72', 'aed05c77b55e69cd67c4a288aad4ffc1d28256d0e6d13a0914852737d1683309'
    ]
# max 0
# salar 1
# ian 2
# niko 3
# mathias 4
# pål 5
# ie-zaim 6
zxc = 7



for i, (model_hash, model) in enumerate(archis.items()):
    
    if i < 100 * zxc:
        continue
    if i >= 100 * (zxc + 1):
        break
    
    if len(hashishish) > 0:
        if model_hash not in hashishish:
            continue
    if len(model["weights"]) == 4 or len(model["weights"]) == 6:
        constraint = "A100|V100"
    else:
        constraint = "A100"
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
    sbatch_cmd = f"sbatch --export=model_hash={hash_value['model_hash']},weights={hash_value['weights']},start_epoch={hash_value['max_epoch']},path={path} --job-name={hash_value['model_hash']} --output=output/{hash_value['model_hash']}.out --constraint={hash_value['constraint']} zzz_slurm/job.slurm"
    subprocess.call(sbatch_cmd.split())