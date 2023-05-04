import json

file_name = "generated_architectures.json"
path = "experiment"

result = []

with open(f"{path}/{file_name}", "r") as f:
    architectures = json.load(f)
    for architecture in architectures:
        zc_scores = {}
        val_acc = {"val_acc" : architectures[architecture]["val_acc"]}
        try:
            zero_cost_scores = architectures[architecture]["zero_cost_scores"]
        except KeyError:
            print(f"{architecture} has no zero cost scores")
            continue
        for zc_score in zero_cost_scores:
            zc_scores[zc_score] = zero_cost_scores[zc_score]["score"]
        
        res = {**zc_scores, **val_acc}
        result.append(res)

with open(f"{path}/data.json", "w") as f:
    print("Writing to file...")
    json.dump(result, f, indent=4, sort_keys=True)
    print("Done!")
    # Print number of architectures in result dict
    print(f"Total architectures generated: {len(result)}")
