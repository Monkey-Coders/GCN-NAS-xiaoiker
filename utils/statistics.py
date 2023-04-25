import json
# Read the file architectures/generated_architectures.json
path = "experiment"

with open(f'{path}/generated_architectures.json', 'r') as f:
    architectures = json.load(f)
    count = 0
    val_acc = []
    for architecture in architectures:
        if "val_acc" in architectures[architecture]:
            count += 1
            val_acc.append(architectures[architecture]["val_acc"])
    val_acc.sort()

    print("We have {} architectures with validation accuracy".format(count))
    print("The average validation accuracy is {}".format(sum(val_acc)/len(val_acc)))
    # find the median of the validation accuracy
    if len(val_acc) % 2 == 0:
        median1 = val_acc[len(val_acc)//2]
        median2 = val_acc[len(val_acc)//2 - 1]
        median = (median1 + median2)/2
    else:
        median = val_acc[len(val_acc)//2]
    print("The median validation accuracy is {}".format(median))
    print("The maximum validation accuracy is {}".format(max(val_acc)))
    print("The minimum validation accuracy is {}".format(min(val_acc)))