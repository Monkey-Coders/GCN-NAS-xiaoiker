import json
# Read the file architectures/generated_architectures.json


with open('architectures_6/generated_architectures.json', 'r') as f:
    architectures = json.load(f)
    count = 0
    val_acc = []
    for architecture in architectures:
        if "val_acc" in architectures[architecture]:
            count += 1
            val_acc.append(architectures[architecture]["val_acc"])

    print("We have {} architectures with validation accuracy".format(count))
    print("The average validation accuracy is {}".format(sum(val_acc)/len(val_acc)))
    print("The maximum validation accuracy is {}".format(max(val_acc)))
    print("The minimum validation accuracy is {}".format(min(val_acc)))