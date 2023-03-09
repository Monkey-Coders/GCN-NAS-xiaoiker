import json

architecture_paths = ["architectures", "architectures_6", "architectures_8", "architectures_10"]

architectures = {}

for architecture_path in architecture_paths:
    with open(f"{architecture_path}/generated_architectures.json", "r") as f:
        architect = json.load(f)
        architectures.update(architect)
        
with open(f'generated_architectures.json', 'w') as f:
    json.dump(architectures, f)
    