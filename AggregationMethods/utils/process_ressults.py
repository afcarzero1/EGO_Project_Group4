"""
Script for processing data of a path
"""
import argparse
import os
import pickle

parser = argparse.ArgumentParser(description="Parser")
parser.add_argument("--path", type=str)

def loadAll(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def showGraphs(accuracy_stats : dict):
    # Create dataframes
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib
    config = accuracy_stats["config"]
    accuracy_stats.pop("config",None)
    #todo : make it plot also the loss function !!
    train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(
        columns={"index": "epochs"})
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))
    sns.lineplot(data=train_val_acc_df, x="epochs", y="value", hue="variable", ax=axes[0]).set_title(
        'Train-Val Accuracy/Epoch')
    plt.show()


if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()
    path = args.path

    if not os.path.exists(path):
        raise FileExistsError("Not existing file")

    file = loadAll(path)
    max_val = 0
    for results in file:
        config=results["config"]
        accuracies_val = results["val"]
        best_val = max(accuracies_val)

        if max_val < best_val:
            max_val = best_val
            max_config = config
            max_results = results

    print(f"The best configuration is {max_config} with accuracy {max_val}")
    showGraphs(max_results)
