import json
import scipy
import matplotlib.pyplot as plt
from ZeroCostFramework.utils.util_functions import get_proxies
base_path = "experiment"

epochs = ["zero_cost_scores_1", "zero_cost_scores_3", "zero_cost_scores_5", "zero_cost_scores_7", "zero_cost_scores_9"]

def validate_architecture(architecture):
    return all(format in architecture for format in ["val_acc", *epochs])


if __name__ == '__main__':
    zero_cost_proxies = ["synflow"] #get_proxies()
    

    with open(f'{base_path}/generated_architectures.json') as f:
        architectures = json.load(f)

    try:
        with open(f'{base_path}/correlations.json') as f:
            correlations = json.load(f)
    except:
        correlations = {}
    
    for epoch in epochs:
        epoch_number = int(epoch.split("_")[-1])
        if epoch not in correlations:
            correlations[epoch] = {}

        for proxy in zero_cost_proxies:
            proxies = []
            val_accs = []
            for architecture in architectures:
                if validate_architecture(architectures[architecture]):
                    if proxy in architectures[architecture][epoch]:
                        proxies.append(architectures[architecture][epoch][proxy]["score"])
                        val_accs.append(architectures[architecture]["val_acc"])

            correlations[epoch][proxy] = scipy.stats.spearmanr(proxies, val_accs)[0]

    with open(f'{base_path}/correlations.json', 'w') as f:
        json.dump(correlations, f)

    for proxy in zero_cost_proxies:
        x = []
        y = []
        for epoch in epochs:
            x.append(int(epoch.split("_")[-1]))
            y.append(correlations[epoch][proxy])
        plt.plot(x, y, label=proxy)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Spearman Correlation")
    # Save
    plt.savefig(f'{base_path}/correlations.png')
    


        
        