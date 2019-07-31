# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

import torch
import torch.nn as nn
from torch.autograd import Variable


class SkipFiltering(nn.Module):

    def __init__(self, N, l_dim):
        """
        Constructing blocks of the skip filtering connections.
        Reference: - https://arxiv.org/abs/1709.00611
                   - https://arxiv.org/abs/1711.01437
        Args :
            N      : (int) Original dimensionallity of the input.
            l_dim  : (int) Dimensionallity of the latent variables.
        """
        super(SkipFiltering, self).__init__()
        print('Constructing Skip-filtering model')
        self._N = N
        self._ldim = l_dim

        self.activation_function = torch.nn.ReLU()

        # Encoder
        self.ih_matrix = nn.Linear(self._N, self._ldim)
        # Decoder
        self.ho_matrix = nn.Linear(self._ldim, self._N)

        # Initialize the weights
        self.initialize_skip_filt()

    def initialize_skip_filt(self):
        """
            Manual weight/bias initialization.
        """
        # Matrices
        nn.init.xavier_normal(self.ih_matrix.weight)
        nn.init.xavier_normal(self.ho_matrix.weight)

        # Biases
        self.ih_matrix.bias.data.zero_()
        self.ho_matrix.bias.data.zero_()
        print('Initialization of the skip-filtering connection(s) model done...')

        return None

    def forward(self, input_x, mask_return=False):

        if torch.has_cudnn:
            x = Variable(torch.from_numpy(input_x).cuda(), requires_grad=True)
        else:
            x = Variable(torch.from_numpy(input_x), requires_grad=True)

        # Encoder
        hl_rep = self.activation_function(self.ih_matrix(x))

        # Decoder
        mask = self.activation_function(self.ho_matrix(hl_rep))

        # Skip-Filtering connection(s)
        y_out = torch.mul(x, mask)

        if mask_return:
            return y_out, x, mask
        else:
            return y_out, x

# EOF
