import os
import json

path = "experiment"

# if False:
#     with open(f"{path}/generated_architectures.json", "r") as f:
#         architectures = json.load(f)

#         # for every hash in the generated architectures
#         # splitt the architectures into sepearate files in a folder
#         # with only the architecture as the name
        
#         for hash in architectures:
#             with open(f"{path}/data/{hash}.json", "w") as f:
#                 json.dump({hash: architectures[hash]}, f, indent=4)
                
if True:
    data = os.listdir(f"{path}/data")
    all_data = {}
    for file in data:
        print(file)
        with open(f"{path}/data/{file}", "r") as f:
            model = json.load(f)
            model_hash = file.split(".")[0]
            all_data[model_hash] = model[model_hash]
    with open(f"{path}/generated_architectures.json", "w") as f:
        f.write(json.dumps(all_data, indent=4))


