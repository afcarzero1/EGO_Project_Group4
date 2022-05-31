"""
Module containing classes for the self-attention and transformer mechanism.
"""

from collections import OrderedDict

from torch import nn
import torch.nn.functional as F
import torch
import numpy as np


class MultiHeadAttentionClassifier(nn.Module):
    r""" Classifier built on top of an encoder like the one described in the paper Attention is all you need.

    """

    def __init__(self, input_size=2048, output_size=8, hidden_sizes=None, encoder_layers=1, attention_heads=1,
                 forward_hidden_size=512, dropout=0.2, mode="all", num_frames=5, add_positional=False,key_size=512):
        super(MultiHeadAttentionClassifier, self).__init__()
        self.mode = mode
        self.squeezing_methods = {"all": self._useAll , "avg" : self._avgSqueezing}
        self.positional_encoder = PositionalEncoder(input_dimension=input_size, maximum_length=6, add=add_positional)
        self.encoder = Encoder(input_size=input_size,
                               number_layers=encoder_layers,
                               number_heads=attention_heads,
                               hidden_size=forward_hidden_size,
                               dropout=dropout,
                               key_size=key_size)
        self.final_attention = MultiHeadAttentionModule(input_size=input_size,
                                                        number_heads=attention_heads,
                                                        dropout=dropout)
        input_size = (input_size * num_frames) if mode == "all" else input_size

        self.average = nn.AvgPool1d(5)

        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128, 64]
        self.sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList()
        for index in range(len(self.sizes) - 1):
            self.layers.append(nn.Linear(self.sizes[index], self.sizes[index + 1]))
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):

        # Add positional encoding
        x = self.positional_encoder(x)
        # Encode the input and use a final attention layer
        x = self.encoder(x)
        x = self.final_attention(x)

        # Use feed forward network to produce the logits
        x = self._squeeze(x)
        x = self._feedForward(x)

        return x

    def _squeeze(self, x: torch.Tensor):
        return self.squeezing_methods[self.mode](x)

    def _useAll(self, x: torch.Tensor):
        batch_size: int = x.size(0)
        return x.view(batch_size, -1)

    def _avgSqueezing(self, x: torch.Tensor):
        batch_size: int = x.size(0)
        # (batch_size,number_of_inputs,input_size)
        return torch.transpose(self.average(torch.transpose(x,-2,-1)),-2,-1)
        # todo : finish this

    def _feedForward(self, x):
        layer: nn.Linear
        for index, layer in enumerate(self.layers):
            # Use layer
            x = layer(x)
            # Apply activation function to all layers except last one.
            if index != (len(self.layers) - 1):
                x = self.relu1(x)
            # Drop-out only the hidden layers
            if index != 0 and index != (len(self.layers) - 1):
                x = self.dropout(x)

        return x


class Encoder(nn.Module):
    def __init__(self, input_size: int = 2048, number_layers=2, dropout: float = 0.3, number_heads=3, hidden_size=1024,
                 key_size=512):
        super(Encoder, self).__init__()
        # Instantiate the attention layers. Stack them in a Module list.
        self.encoder_layers = nn.ModuleList(
            [MultiHeadAttentionModule(input_size=input_size,
                                      number_heads=number_heads,
                                      residual=True,
                                      key_size=key_size,
                                      value_size=512,
                                      dropout=dropout) for i in range(number_layers)]

        )
        # Instantiate feed forward layers
        self.feed_forward_layers = nn.ModuleList(
            [EncoderFeedForwardNormalized(input_features=input_size,
                                          hidden_size=hidden_size,
                                          dropout=dropout)
             for i in range(number_layers)])

    def forward(self, x):
        for encoder, feed_forward in zip(self.encoder_layers, self.feed_forward_layers):
            x = encoder(x)
            x = feed_forward(x)
        return x


class MultiHeadAttentionModule(nn.Module):
    r"""This is a module implementing a slightly modified version of the multi-head attention layer described in the
    paper Attention is all you need.

    Attributes:

    """

    def __init__(self, input_size=2048, number_heads=1, residual=True, key_size=512, value_size=None,
                 dropout: float = 0.3):
        r""" Initializer of the multi-head attention module

        In this function the base modules of the architecture are defined. We have
            1) Linear layers mapping key,query and value
            2) Attention module for the attention matrix
        """
        # Call super class initializer
        super(MultiHeadAttentionModule, self).__init__()

        # Define internal variable with the input size
        self.input_size: int = input_size
        self.key_size: int = key_size
        self.value_size: int = key_size if value_size is None else value_size
        self.number_heads = number_heads
        self.residual = residual
        # Linear layers for computing the query,key and value. They compute them for each head of the model
        self.query_linear = nn.Linear(input_size, key_size * number_heads, bias=False)
        self.key_linear = nn.Linear(input_size, key_size * number_heads, bias=False)
        self.value_linear = nn.Linear(input_size,
                                      value_size * number_heads if value_size is not None else key_size * number_heads,
                                      bias=False)

        # Module for computing the attention
        self.attention_module = AttentionModule()

        self.linear_out = nn.Linear(self.number_heads * self.value_size, input_size)
        self.normalizer = nn.LayerNorm(input_size)
        # Regularization (dropout)
        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax()

    def forward(self, x):
        r""" Function overriding method in nn Module. It computes the multi-head attention result
        It applies

        Args:
            x: Input tensor to process. The accepted dimension is (batch_size,num_inputs_per_record,input_size)

        Returns:
            y : Processed tensor

        """
        # Get the dimensions
        batch_size = x.size(0)
        frames_number = x.size(1)

        # Compute query, key and value as output of linear layers. Resize for correct dimensionality
        # (batch_size, number_frames,number_heads,key/value dimensionality)
        query: torch.Tensor = self.query_linear(x).view(batch_size, frames_number, self.number_heads, self.key_size)
        key: torch.Tensor = self.key_linear(x).view(batch_size, frames_number, self.number_heads, self.key_size)
        value: torch.Tensor = self.value_linear(x).view(batch_size, frames_number, self.number_heads, self.value_size)

        # Compute the attention matrix
        # (batch_size,num_heads,input_size,input_size)
        attention_matrix = self.attention_module(query, key)

        # Compute the matrix output size
        # (batch_size,num_heads,input_size,value_size)
        matrix_mul_out = torch.matmul(attention_matrix, torch.transpose(value, -2, -3))

        # Concatenate all heads output. For this just merge the last dimensions
        # (batch_size,number_frames,number_heads*value_size)
        matrix_mul_out = torch.transpose(matrix_mul_out, -2, -3). \
            contiguous().view(batch_size, frames_number, (self.number_heads * self.value_size))

        # Add layer out and use dropout
        matrix_mul_out = self.linear_out(matrix_mul_out)
        matrix_mul_out = self.dropout(matrix_mul_out)

        # Add residual if specifies by the model
        x = matrix_mul_out + (x if self.residual else 0)
        x = self.normalizer(x)

        return x


class AttentionModule(nn.Module):
    r""" Module for computing the attention mechanism. This module computes the attention matrix.

    The computation is done as follows:
    1 ) Matrix multiplication between query and key
    2 ) Scalation factor
    3 ) Softmax applied by rows

    Args:
        dot_product_type (str): It is the type of dot product we are doing.

    """

    def __init__(self, dot_product_type="scaled"):
        super(AttentionModule, self).__init__()

        # Dot products
        self.dot_products = {"scaled": self._scaledDotProduct}
        self.dot_product_type = dot_product_type

    def forward(self, query, key):
        r""" Compute the attention matrix obtained combining query and key.

        This function implements the mechanism described in the Attention is all you need paper.
            1) Matrix multiplication between key and query
            2) Softmax function applied to matrix rows
        Args:
            query (torch.Tensor): This is the query matrix. The admitted dimension is (batch_size,num_frames,num_heads,input_size)
            key (torch.Tensor): This is the key matrix. The admitted dimension is (batch_size,num_frames,num_heads,input_size)

        Returns:
            attention (torch.Tensor) : Attention tensor
        """
        # Transpose for having (batch_size,num_heads,num_frames,input_size)
        query = torch.transpose(query, -2, -3)
        key = torch.transpose(key, -2, -3)
        # Compute attention matrix
        attention = self.dot_products[self.dot_product_type](query, key)
        # Compute the softmax along the rows for having numbers between 0 and 1
        attention = F.softmax(attention, dim=-1)
        return attention

    def _scaledDotProduct(self, query: torch.Tensor, key: torch.Tensor):
        r""" Private function implementing the scaled dot product between query and key for computing attention matrix.

        This is the original implementation of the scaled dot product used in the Attention is all you need paper.
        Args:
            query (torch.Tensor) : Query passed to vector. The admitted dimension is
            key (torch.Tensor) : Key passed to the vector. The admitted dimension is

        Returns:
            attention (torch.Tensor) : Tensor with the attention matrix. The dimension is
            (num_heads,input_size,input_size)
        """
        attention = torch.matmul(query, torch.transpose(key, -2, -1))
        # Divide by the dimensionality of the key
        attention = attention / np.sqrt(key.size(dim=-1))
        return attention


class EncoderFeedForwardNormalized(nn.Module):
    def __init__(self, input_features=2048, hidden_size=1024, dropout=0.2):
        super(EncoderFeedForwardNormalized, self).__init__()

        self.linear1 = nn.Linear(in_features=input_features, out_features=hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=input_features)
        self.dropout = nn.Dropout(dropout)
        self.normalizator = nn.LayerNorm(input_features)

    def forward(self, x):
        # Add residual and pass thorugh a simple feed forward network.
        x = self.dropout(self.linear2(self.relu(self.linear1(x)))) + x
        x = self.normalizator(x)
        return x


class PositionalEncoder(nn.Module):
    def __init__(self, input_dimension: int, maximum_length: int = 20, add=True):
        super(PositionalEncoder, self).__init__()

        # todo : change this parameter dynamically when is necessary
        # todo : It specifies the maximum number
        self.maximum_length: int = maximum_length
        self.input_dimension = input_dimension

        if add:
            self.matrix = self._buildMatrix()
        else:
            # In case it is deactivated put a zero matrix
            self.matrix = torch.zeros((self.maximum_length, self.input_dimension))

        if torch.cuda.is_available():
            self.matrix = self.matrix.cuda()

    def _buildMatrix(self):

        # Get numerator
        position_vector: torch.Tensor = torch.arange(0, self.maximum_length, 1)

        # Get Divisor
        input_index: torch.Tensor = torch.arange(0, self.input_dimension, 1) * 2 / self.input_dimension
        input_index = torch.pow(1000, input_index)
        input_index = input_index.pow(-1)

        position_vector = position_vector.view(self.maximum_length, 1).float()
        input_index = input_index.view(1, self.input_dimension).float()
        matrix = torch.matmul(position_vector, input_index)

        matrix[0::2] = torch.sin(matrix[0::2])
        matrix[1::2] = torch.cos(matrix[1::2])
        return matrix

    def forward(self, x):
        number_records = x.size(-2)
        input_size = x.size(-1)
        return x + self.matrix[0:number_records, 0:input_size]


if __name__ == '__main__':
    instance = PositionalEncoder(input_dimension=10, maximum_length=5)
    print(instance.matrix)

    test = torch.zeros((5, 10))
    result = instance(test)
    print(result)
