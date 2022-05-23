import json
import sys

import torch
import pickle
import pandas as pd
import os


def multiclassAccuracy(y_pred, y_test):
    """
    Function for computing the multiclass accuracy
    Args:
        y_pred: predicton donte by the model
        y_test: ground truth

    Returns:
        The accuracy of the predictions

    """
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


def explorePkl():
    """
    Function for exploring the pickle files
    Returns:

    """
    global args

    file_path = "/home/andres/MLDL/Pre-extracted/Flow/ek_i3d/D2-D2_train.pkl"

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    flow = data['features']['Flow']
    narration_ids = data['narration_ids']
    print(data)

    file_path2 = "/home/andres/MLDL/EGO_Project_Group4/train_val/D2_train.pkl"
    dataDF: pd.DataFrame = pd.read_pickle(file_path2)
    print(dataDF)


def saveMetrics(metrics, file_name, path, args):
    # Check if it exists
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise NotADirectoryError(f"{path} is not a directory")
    else:
        # Create the folder
        os.mkdir(path)

    pkl_file_name = file_name + ".pkl"
    pkl_file_name = os.path.join(path, pkl_file_name)
    with open(pkl_file_name, "ab+") as out:
        metrics["config"] = args["config"]
        pickle.dump(metrics, out)

    json_file_name = file_name + ".json"
    json_file_name = os.path.join(path, json_file_name)
    with open(json_file_name, "a+") as out:
        metrics["config"] = args["config"]
        json.dump(metrics, out)


def getFileName(args=None) -> str:
    """
    Function for generating the file name of the file where to save the results of the execution.
    Args:
        args (dict): Arguments of the program

    Returns:
        file_name(str) : File name
    """
    if args == None:
        model_name: str = "UNKNOWN"
    else:
        model_name: str = args["temporal_aggregator"]

    dt_string = getTimestamp()

    return dt_string + '_' + model_name + '_' + args["model"] + '_' + args["modality"]


def getFolderName(args):
    results_path = os.path.join(args["base_path"], args["results_location"])
    timestamp_string = getTimestamp()
    results_path = os.path.join(results_path, timestamp_string)
    return results_path


def getBasePath(file: str):
    return os.path.dirname(os.path.realpath(file))


def getTimestamp():
    from datetime import datetime

    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    return dt_string


if __name__ == '__main__':
    args = {}
    args["results_location"] = "RESULTS_AGGREGATION"
    args["base_path"] = getBasePath(__file__)
    saveMetrics(None, "metrics_test", args)
