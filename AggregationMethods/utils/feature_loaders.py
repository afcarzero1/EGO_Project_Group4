from torch.utils.data import DataLoader, Dataset
import sys
import pickle
import pandas as pd
import numpy as np
import torch


class FeaturesDataset(Dataset):
    r""" Class implementing the dataset of features associated to a specific domain (kitchen)

    """

    def __init__(self, model="i3d", modality="Flow", shift="D1-D1", train=True):
        super(FeaturesDataset, self).__init__()

        # This is a sample path we want to get!
        # /home/andres/MLDL/Pre-extracted/Flow/ek_i3d/D2-D2_train.pkl"

        # This is the base path we will use for getting the sub-paths
        base_path = "/home/andres/MLDL/Pre-extracted/"  # todo: re-define in more general way to work on another PC

        # Build the sub-path we are interested in
        base_path += modality + '/'
        base_path += "ek_" + model.lower() + '/'
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
        same record.
        Args:
            index: It is the index of the record to retrieve

        Returns:
            Feature vector and label of the required record

        """
        if index < 0 or index > len(self.labels):
            raise IndexError
        # Cast to torch tensor from numpy array to use in torch models.
        torch_tensor = torch.from_numpy(self.features[index])
        #todo : For efficiency it is better to do this transformations when initializing the dataset, change it

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

