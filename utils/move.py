import json
from distutils.dir_util import copy_tree
import os
import numpy as np
from tqdm import tqdm
import yaml

from_path = "architectures_8"
to_path = "experiment"

def has_nan(d):
    for k, v in d.items():
        if isinstance(v, dict):
            if has_nan(v):
                return True
        elif isinstance(v, float) and float(v) == float('nan'):
            return True
    return False

def fix_config(to_path, model_hash):
    with open(f"{to_path}/run/{model_hash}/work_dir/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    config["work_dir"] = f"experiment/run/{model_hash}/work_dir"
    config["model_saved_name"] = f"experiment/run/{model_hash}/runs"
    config["config"] = f"experiment/configs/{model_hash}.yaml"
    
    with open(f"{to_path}/run/{model_hash}/work_dir/config.yaml", "w") as f:
        yaml.safe_dump(config, f)

with open(f"{from_path}/generated_architectures.json", "r") as f:
    from_architectures = json.load(f)
with open(f"{to_path}/generated_architectures.json", "r") as f:
    to_architectures = json.load(f)
    
for model_hash, model in tqdm(from_architectures.copy().items()):
    if "val_acc" in model:
        if model_hash not in to_architectures:    
            if has_nan(model):
                continue
            try:
                copy_tree(f"{from_path}/run/{model_hash}", f"{to_path}/run/{model_hash}")
                fix_config(to_path, model_hash)
                to_architectures[model_hash] = model
                del from_architectures[model_hash]
            except:
                continue

with open(f"{from_path}/generated_architectures.json", "w") as f:
    json.dump(from_architectures, f)
with open(f"{to_path}/generated_architectures.json", "w") as f:
    json.dump(to_architectures, f)