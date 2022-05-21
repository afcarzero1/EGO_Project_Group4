"""
This is a script for training a classifier based on pre-extracted featres from the videos.
"""
import sys
from tqdm import tqdm
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

    # Instantiate the dataset and the data loader and the model.

    dataset_train = FeaturesDataset()
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=100)  # For now define batch size = 1
    model = NeuralNetAvgPooling()

    # Use the cross entrpy loss and the Adam optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Do the training

    # This is the number of times we go through the entire data!
    epochs = 500
    print("[GENERAL] The model used is:\n")
    print(model)

    # Initialize statistics about training
    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }

    print("[GENERAL] Starting training")
    print("[GENERAL] Moving model to GPU")
    model.to(device)

    for e in tqdm(range(1, epochs + 1)):
        # Set up model for training
        model.train()

        train_epoch_loss = 0
        train_epoch_acc = 0

        for X_train_batch, y_train_batch in dataloader_train:
            # Move data to GPU
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            # Set the gradient to 0
            optimizer.zero_grad()
            # Apply model to the batch. Forward propagation
            y_train_pred = model(X_train_batch)


            batch_size = X_train_batch.size()[0]

            # Resize for the cross entropy loss. todo: add here constant num_classes
            y_train_pred = torch.reshape(y_train_pred, (batch_size, 8))
            y_train_batch = torch.reshape(y_train_batch, (batch_size,))
            # Compute loss
            train_loss = criterion(y_train_pred, y_train_batch)
            train_accuracy = multiclassAccuracy(y_train_pred, y_train_batch)

            # Backpropagate the gradient. Take a step in "correct" direction
            train_loss.backward()  # Accumulates gradients
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_accuracy.item()

        # Append the average batch loss
        loss_stats["train"].append(train_epoch_loss/len(dataloader_train))
        accuracy_stats["train"].append(train_epoch_acc/len(dataloader_train))
        #Append the average batch accuracy

        # print message
        if e % 10 ==0:
            tqdm.write(
            f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(dataloader_train):.5f} | Train Acc: {train_epoch_acc / len(dataloader_train):.3f}')


def minimalTest():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[GENERAL] Using {device}")
    # Instantiate the dataset, dataloader and model
    dataset = FeaturesDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    model = NeuralNetAvgPooling()
    # Take a record from the dataset
    data_test = dataset[0]
    features: torch.Tensor = data_test[0]
    label = data_test[0]

    print(f"The dimension is {features.size()}")
    print(f"The label is {label}")

    # Apply the model 1D average pooling. The pooling is done
    test_tensor = torch.Tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    test_tensor_transposed = torch.transpose(test_tensor, 0, 1)
    test_pooler = nn.AvgPool1d(2)
    test_output = test_pooler(test_tensor)
    test_output_transposed = test_pooler(test_tensor_transposed)
    print(test_tensor)
    print(test_output)
    print(test_tensor_transposed)
    print(test_output_transposed)

    out = model(features.float())
    print(out)


def minimalTrain():
    """
    Script for testing with CPU. Initial test on the Average Pooling
    Returns:

    """
    dataset = FeaturesDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=1)  # For now define batch size = 1
    model = NeuralNetAvgPooling()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Do the training

    # This is the number of times we go through the entire data!
    epochs = 500

    print(model)

    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }

    print("[GENERAL] Starting training")

    for e in tqdm(range(1, epochs + 1)):
        # Set up model for training
        model.train()

        train_epoch_loss = 0
        train_epoch_acc = 0

        for X_train_batch, y_train_batch in dataloader:
            # todo : Add move to GPU of data :)

            # Set the gradient to 0
            optimizer.zero_grad()
            # Apply model to the batch. Forward propagation
            y_train_pred = model(X_train_batch)

            # Compute loss

            y_train_pred = y_train_pred[0]
            y_train_batch = y_train_batch[0]

            train_loss = criterion(y_train_pred, y_train_batch)
            train_accuracy = multiclassAccuracy(y_train_pred, y_train_batch)

            # Backpropagate the gradient. Take a step in "correct" direction
            train_loss.backward()  # Accumulates gradients
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_accuracy.item()


class NeuralNetAvgPooling(nn.Module):
    r"""This is the classifier to apply after the feature extractor.

    Attributes:
        layers([]) : List of neural network layers
        sizes ([int]) : List with dimensionality of layers
    """

    def __init__(self, input_size=1024, output_size=8, hidden_sizes=None):
        """
        This is the initialization function of the classifier. It is a multiclass
        classifier with multiple hidden layers
        Args:
            input_size (int): Dimensionality of input features
            hidden_sizes([int]): List of integers with hidden layers dimensionality
            output_size(int): Dimensionality of output
            dropout(nn.Dropout): The dropout
        """
        # Call super class intitializer
        super(NeuralNetAvgPooling, self).__init__()
        # Use default sizes
        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128, 64]

        # Instantiate all layers
        self.sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList()
        self.batch_normalizer = nn.ModuleList()

        self.average = nn.AvgPool1d(5)

        # Create all layers of linear and batch normalization
        for index in range(len(self.sizes) - 1):
            self.layers.append(nn.Linear(self.sizes[index], self.sizes[index + 1]))
            self.batch_normalizer.append(nn.BatchNorm1d(self.sizes[index + 1]))

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
        # Apply to the tensor (n_features,n_clips)
        x = self.average(x)
        # Transpose to have (1,n_features)
        x = torch.transpose(x, -2, -1)

        # Apply all the layers to the input

        layer: nn.Linear
        for index, layer in enumerate(self.layers):

            x = layer(x)
            # x = self.batch_normalizer[index](x) todo: fix this !! How do I add batch normalization? Where to define batch size?
            # todo : Add batch normalization for faster convergence. How does it work?

            # Apply activation function to all layers except last one.
            if index != (len(self.layers) - 1):
                x = self.relu1(x)
            # Drop-out only the hidden layers
            if index != 0 and index != (len(self.layers) - 1):
                x = self.dropout(x)
        return x


class FeaturesDataset(Dataset):
    r""" Class implementing the dataset of features associates to a specific domain.

    """

    def __init__(self, model="i3d", modality="Flow", shift="D1-D1", train=True):
        super(FeaturesDataset, self).__init__()

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

    def __getitem__(self, index: int):
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
        # Cast to torch tensor from numpy array to use in torch models.
        torch_tensor = torch.from_numpy(self.features[index])
        # Transpose to have as the first dimension the clips
        torch_tensor = torch.transpose(torch_tensor, 0, 1).float()

        label_vector = torch.tensor([self.labels[index]], dtype=torch.long)
        # label_vector = torch.zeros(8)
        # label_vector[self.labels[index]]=1

        return torch_tensor, label_vector.long()

    def __len__(self):
        r""" Overriding of in-built function for length of the dataset.
        Returns: The number of records present in this dataset.
        """
        return len(self.labels)


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
    # minimalTest()
    # minimalTrain()
    main()
