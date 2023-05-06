import json
import os
# Read the file architectures/generated_architectures.json
path = "experiment"

with open(f'{path}/generated_architectures.json') as f:
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

    # Calculate the minimum, average and maximum training time in seconds, but converted to hours
    training_time = []
    for architecture in architectures:
        if "time" in architectures[architecture]:
            training_time.append(architectures[architecture]["time"])
    training_time.sort()
    print("The average training time is {} hours".format(sum(training_time)/len(training_time)/3600))
    print("The maximum training time is {} hours".format(max(training_time)/3600))
    print("The minimum training time is {} hours".format(min(training_time)/3600))

    # Find the average number of layers, minimum and maximum number of layers
    layers = []
    for architecture in architectures:
        layers.append(len(architectures[architecture]["weights"]))
    
    print(f"Archi with 10 layers: {len([l for l in layers if l == 10])}")
    print(f"Archi with 8 layers: {len([l for l in layers if l == 8])}")
    print(f"Archi with 6 layers: {len([l for l in layers if l == 6])}")
    print(f"Archi with 4 layers: {len([l for l in layers if l == 4])}")
    print("The average number of layers is {}".format(sum(layers)/len(layers)))
    print("The maximum number of layers is {}".format(max(layers)))
    print("The minimum number of layers is {}".format(min(layers)))

    # Find minimum, maximum and average number of parameters
            