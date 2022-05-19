"""
This is a script for training a classifier based on pre-extracted featres from the videos.
"""
import sys

from torch.nn import Linear

from utils.args import parser

args = parser.parse_args()
import pickle
import pandas as pd
import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim


def main():
    # Get arguments # todo : Use them to define the model
    global args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[GENERAL] Using {device}")
    # Instantiate the dataset
    dataset = featuresDataset()

    dataloader = DataLoader(dataset=dataset,batch_size=1)

    model = NeuralNet()

    #todo : Look that everything works in GPU. How to make everything in GPU for fast training?

    # Mover everything to GPU
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.01)


    # Do the training

    epochs = 500
    





class NeuralNet(nn.Module):
    r"""This is the classifier to apply after the feature extractor.

    """

    def __init__(self, input_size=1024, output_size=8, hidden_sizes=None):
        """
        This is the initialization function of the classifier. It is a multiclass
        classifier with multiple hidden layers
        Args:
            input_size:
            hidden_sizes:
            output_size:
        """
        # Call super class intitializer
        super(NeuralNet, self).__init__()
        # Use default sizes
        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128, 64]

        # Instantiate all layers
        self.sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = []

        #self.aggregation = acergae pooling or trn
        for index in range(len(self.sizes) - 1):
            self.layers.append(nn.Linear(self.sizes[index], self.sizes[index + 1]))

        # Instantiate activation function and regularization parameters
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        r""" Function overriding method in nn Module.
        Args:
            x: Input tensor to process.

        Returns:
            y : Processed tensor

        """
        # Apply all the layers to the input
        layer: nn.Linear
        for index, layer in enumerate(self.layers):
            x = layer(x)
            # todo :Add batch normalization. Is it useful?
            x = self.relu1(x)
            # Drop-out only the hidden layers
            if index != 0 and index != (len(self.layers) - 1):
                x = self.dropout(x)
        return x


class featuresDataset(Dataset):
    r""" Class implementing the dataset of features associates to a specific domain.

    """
    def __init__(self, model="i3d", modality="Flow", shift="D1-D1", train=True):
        super(featuresDataset, self).__init__()

        # This is a sample path we want to get!
        # /home/andres/MLDL/Pre-extracted/Flow/ek_i3d/D2-D2_train.pkl"

        # This is the base path we will use for getting the sub-paths
        base_path = "/home/andres/MLDL/Pre-extracted/"  # todo: re-define in more general way to work on another PC

        # Build the sub-path we are interested in
        base_path += modality + '/'
        base_path += "ek_" + model + '/'
        base_path += shift + '_' + ("train" if train else "test") + ".pkl"

        # Open the pickle file containing the features
        try:
            with open(base_path, 'rb') as f:
                self.raw_data = pickle.load(f)
        except IOError:
            print(f"File not found : {base_path}. Try to check the file exists and the path is correct.")
            sys.exit()

        # Extract the features. The dimension is (num_records,num_clips,feature_dimension)
        self.features: np.ndarray = self.raw_data['features'][modality]
        self.narration_ids = self.raw_data['narration_ids']

        # Extract the labels from other file
        labels_path = "/home/andres/MLDL/EGO_Project_Group4/train_val/"
        # Take the labels of the "validation" domain.
        labels_path += shift.split("-")[1] + "_" + ("train" if train else "test") + ".pkl"
        labelsDF: pd.DataFrame = pd.read_pickle(labels_path)

        ids = labelsDF["uid"]
        # todo : add check that uid correspond to the narration id
        self.labels = labelsDF["verb_class"]

    def __getitem__(self, index : int):
        """ Overriding of in-built function for getting an item. Returns record of dataset.

        This function returns a vector of features with dimensionality of 1024. It corresponds to 5 clips taken from the
        same record. # FIXME : Is it correct to return the 5 clips or should I return only one?. How to handle this?
        Args:
            index: It is the index of the record to retrieve

        Returns:
            Feature vector and label of the required record

        """
        if index < 0 or index > len(self.labels):
            raise IndexError
        return self.features[index], self.labels[index]

    def __len__(self):
        r""" Overriding of in-built function for length of the dataset.
        Returns: The number of records present in this dataset.
        """
        return len(self.labels)


# 5 x 1024

# SEARCH TRN FOR AGGREGATION !! (look online).



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


if __name__ == '__main__':
    main()
