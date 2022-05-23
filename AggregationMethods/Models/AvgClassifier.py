import torch
from torch import nn

class NeuralNetAvgPooling(nn.Module):
    r"""This is the classifier to apply after the feature extractor.

    Attributes:
        layers([]) : List of neural network layers
        sizes ([int]) : List with dimensionality of layers
    """

    def __init__(self, input_size=1024, output_size=8, hidden_sizes=None, dropout: float = 0.3):
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
        self.dropout = nn.Dropout(p=dropout)

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

