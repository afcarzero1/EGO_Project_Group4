"""
Script for processing data of a path
"""
import argparse
import os
import pathlib
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description="Parser")
parser.add_argument("--path", type=str)
parser.add_argument("--metric",type=str,default="early")
parser.add_argument("--patience",type=int,default=3)
def main():
    args = parser.parse_args()
    path = args.path
    analyzer = ResultsAnalyzer(path, metric=args.metric,patience=args.patience)
    configuration_results = analyzer.analyze()

    df = analyzer.transformIntoDataFrame(configuration_results)
    print(df)
    df.to_csv(os.path.join(path,args.metric+"ConfigurationResults.csv"))

    average_results = df.mean(axis=1)
    print("[GENERAL] Average results are:")
    print(average_results)
    print("[GENERAL] The best configuration and the respective index are:")
    average_results.to_csv(os.path.join(path,args.metric+"AverageConfigurationResults.csv"))
    best_configuration = average_results.idxmax()
    print(average_results.max(), best_configuration)

    data_best = analyzer.findDataConfiguration(best_configuration)

    analyzer.plotData(data_best,best_configuration)


def oldmain():
    # Parse args
    args = parser.parse_args()
    path = args.path

    if not os.path.exists(path):
        raise FileExistsError("Not existing file")

    file = loadAll(path)
    max_val = 0
    for results in file:
        config = results["config"]
        accuracies_val = results["val"]
        best_val = max(accuracies_val)

        if max_val < best_val:
            max_val = best_val
            max_config = config
            max_results = results

    print(f"The best configuration is {max_config} with accuracy {max_val}")
    showGraphs(max_results)


def loadAll(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def showGraphs(accuracy_stats: dict):
    # Create dataframes
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib
    config = accuracy_stats["config"]
    accuracy_stats.pop("config", None)
    # todo : make it plot also the loss function !!
    train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(
        columns={"index": "epochs"})
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))
    sns.lineplot(data=train_val_acc_df, x="epochs", y="value", hue="variable", ax=axes[0]).set_title(
        'Train-Val Accuracy/Epoch')
    plt.show()


class ResultsAnalyzer():
    """

    """

    def __init__(self, base_path, metric="max",patience=3):
        pass

        self.folder: pathlib.Path = Path(base_path)

        # Get all the pickle files in this folder (accuracies in all domains)
        self.files = self.folder.glob("*accuracies.pkl")
        print(len(list(self.files)))

        # Initialize metrics
        self.patience=patience
        self.metric = metric
        self.metrics = {"last": self.lastMetric, "max": self.bestMetric, "early": self.earlyMetric}
        if metric not in self.metrics:
            raise ValueError(f"Not known {metric}")

    def analyze(self, test_domain=True):
        """
        Analyze the results
        Args:
            test_domain(bool):

        Returns:

        """
        if test_domain:
            self.list_id = "val"
        else:
            self.list_id = "train"

        configurations = {}  # Dictionary with metrics for all configurations
        for domain_path in self.folder.glob("*accuracies.pkl"):
            file_name = os.path.basename(os.path.normpath(domain_path))
            print(f"[GENERAL] Analyzing the file {file_name}")
            file = loadAll(domain_path)
            for results in file:
                # Get the metrics to analyze
                result_metrics: list = results[self.list_id]
                best_val: float = self.metrics[self.metric](result_metrics)

                # Get the configuration to which to associate the result
                config: () = results["config"]
                key_config = self._getKey(config)
                if not key_config in configurations:
                    configurations[key_config] = [(best_val, file_name)]
                else:
                    configurations[key_config] += [(best_val, file_name)]
        return configurations

    def transformIntoDataFrame(self, configurations):
        domain_results = next(iter(configurations.values()))
        column_names = []
        for domain in domain_results:
            column_names.append(domain[1])
        for config, results in configurations.items():
            new_list = []
            for result in results:
                new_list.append(result[0])
            configurations[config] = new_list

        df = pd.DataFrame.from_dict(configurations, orient='index')
        df.columns = column_names
        return df

    def findDataConfiguration(self, configuration: frozenset):
        data = {}
        for domain_path in self.folder.glob("*accuracies.pkl"):
            file_name = os.path.basename(os.path.normpath(domain_path))
            print(f"[GENERAL] Analyzing the file {file_name}")
            file = loadAll(domain_path)
            for results in file:
                # Get the configuration to which to associate the result
                config: () = results["config"]
                key_config = self._getKey(config)
                if key_config == configuration:
                    # In case of match
                    result_metrics: (list, list) = (results["val"], results["train"])
                    data[file_name] = result_metrics

        return data

    def plotData(self, data,best_configuration=None):
        best_configuration = "" if best_configuration is None else best_configuration
        best_configuration = str(best_configuration).replace("frozenset","")
        fig, axs = plt.subplots(1, int(len(data)), sharex=True, sharey=True)
        fig.suptitle(f"Accuracies best configuration\n{best_configuration}")
        fig.text(0.5, 0.04, 'Epochs', ha='center')
        fig.text(0.04, 0.5, 'Accuracy [%]', va='center', rotation='vertical')
        for i, (file_name, result_metrics) in enumerate(data.items()):
            epochs = np.arange(0, len(result_metrics[0]))
            val_metric = result_metrics[0]
            train_metric = result_metrics[1]
            axs[i].plot(epochs, val_metric, label="validation")
            axs[i].plot(epochs, train_metric, label="train")
            axs[i].legend(loc='upper right')

        plt.show()

    def _getKey(self, config):
        external_parameters = config[0]
        internal_parameters = config[1]
        configuration_dictionary = external_parameters | internal_parameters
        return frozenset(configuration_dictionary.items())

    def lastMetric(self, vec1):
        return vec1[-1]

    def bestMetric(self, vec1):
        return max(vec1)

    def earlyMetric(self, vec1):
        last_val = np.inf
        patience = self.patience
        trigger_times = 0
        for i, val in enumerate(vec1):
            if val < last_val:
                trigger_times += 1
                if trigger_times > patience:
                    return vec1[i - patience]
            else:
                trigger_times = 0

            last_val = val


if __name__ == '__main__':
    main()
