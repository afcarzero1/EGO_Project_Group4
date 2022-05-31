import torch
from Models.MultiHeadAttention import MultiHeadAttentionModule
from Models.AvgClassifier import NeuralNetAvgPooling
import torch.nn as nn
import torch.nn.functional as F


class AttentionBaseline(nn.Module):
    def __init__(self, dropout=0.1, input_size=2048,number_heads=1,hidden_sizes=None):
        super(AttentionBaseline, self).__init__()
        # Define default parameter
        if hidden_sizes is None:
            hidden_sizes = [512]

        # Define the architecture

        self.attention = MultiHeadAttentionModule(number_heads=number_heads,
                                                  input_size=input_size)

        # Define layer after

        self.avg_feed = NeuralNetAvgPooling(input_size=input_size,
                                            hidden_sizes=hidden_sizes,
                                            dropout=dropout)

    def forward(self, x):
        """
        Overriding of the forward function. In this function we will use the entire input for computing the attention
        matrix and we will proceed to use it for the classification
        Args:
            x:

        Returns:

        """
        attention_matrix = self.attention(x)

        attention_matrix = F.softmax(attention_matrix, dim=-1)
        # ElementWise multiplication
        x = torch.mul(attention_matrix, x)

        # Do a regular average pooling and feed forward
        x = self.avg_feed(torch.transpose(x,-1,-2))

        return x
