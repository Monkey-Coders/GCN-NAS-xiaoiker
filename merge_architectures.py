import json

from ZeroCostFramework.utils.util_functions import get_proxies



save_folder = "experiment"

if False:
    architecture_paths = ["architectures", "architectures_6", "architectures_8", "architectures_10"]
    
    architectures = {}
    
    for architecture_path in architecture_paths:
        with open(f"{architecture_path}/generated_architectures.json", "r") as f:
            architect = json.load(f)
            architectures.update(architect)
            
    with open(f'{save_folder}/generated_architectures.json', 'w') as f:
        json.dump(architectures, f)
    
if False:
    new_architectures = {}
    with open(f"{save_folder}/generated_architectures.json", "r") as f:
        architectures = json.load(f)
        for (key, value) in architectures.items():
            if "val_acc" not in value:
                continue
            # nan = False
            # for (_key, new_value) in value["zero_cost_scores"].items():
            #     if new_value["score"] == 'nan':
            #         nan = True
            #         break
            # if nan:
            #     continue
            new_architectures[key] = value
            
    with open(f'{save_folder}/generated_architectures.json', 'w') as f:
        json.dump(new_architectures, f)
    
if False:
    new_architectures = {}
    with open(f"{save_folder}/generated_architectures.json", "r") as f:
        architectures = json.load(f)
        for (key, value) in architectures.items():
            nan = False
            for (_key, new_value) in value["zero_cost_scores"].items():
                if new_value["score"] == 'nan':
                    nan = True
                    break
            if nan:
                del value["zero_cost_scores"]
            new_architectures[key] = value
            
    with open(f'{save_folder}/generated_architectures.json', 'w') as f:
        json.dump(new_architectures, f)
        
if True:
    new_architectures = {}
    proxies = get_proxies()
    with open(f"{save_folder}/generated_architectures.json", "r") as f:
        architectures = json.load(f)
        for (key, value) in architectures.items():
            nan = False
            if "zero_cost_scores" in value:
                keys = [new_key for (new_key, new_value) in value["zero_cost_scores"].items()]
                for p in proxies: 
                    if p not in keys:
                        nan = True
                    
            if nan:
                del value["zero_cost_scores"]
            new_architectures[key] = value
            
    with open(f'{save_folder}/generated_architectures.json', 'w') as f:
        json.dump(new_architectures, f)
        
