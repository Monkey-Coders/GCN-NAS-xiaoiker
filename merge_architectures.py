import json


save_folder = "experiment"

if True:
    architecture_paths = ["architectures", "architectures_6", "architectures_8", "architectures_10"]
    
    architectures = {}

    with open(f'{save_folder}/generated_architectures.json', 'r') as f:
        architectures = json.load(f)
    
    for architecture_path in architecture_paths:
        with open(f"{architecture_path}/generated_architectures.json", "r") as f:
            architect_x = json.load(f)
            for (key, value) in architect_x.items():
                if key in architectures:
                    continue
                if "val_acc" not in value:
                    continue
                architectures[key] = value
            
    with open(f'{save_folder}/generated_architectures.json', 'w') as f:
        json.dump(architectures, f)
