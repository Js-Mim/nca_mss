# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

import torch
import torch.nn as nn
from torch.autograd import Variable


class DFFN(nn.Module):

    def __init__(self, N, l_dim):
        """
        Constructing blocks for a two-layer FFN.
        Args :
            N      : (int) Original dimensionallity of the input.
            l_dim  : (int) Dimensionallity of the latent variables.
        """
        super(DFFN, self).__init__()
        print('Constructing 2-FFN')
        self._N = N
        self._ldim = l_dim

        self.activation_function = torch.nn.ReLU()

        # Encoder
        self.ih_matrix = nn.Linear(self._N, self._ldim, bias=True)
        # Decoder
        self.ho_matrix = nn.Linear(self._ldim, self._N, bias=True)

        # Initialize the weights
        self.initialize_ffn()

    def initialize_ffn(self):
        """
            Manual weight/bias initialization.
        """
        # Matrices
        nn.init.xavier_normal(self.ih_matrix.weight)
        nn.init.xavier_normal(self.ho_matrix.weight)

        print('Initialization of the FFN done...')

        return None

    def forward(self, input_x):

        if torch.has_cudnn:
            x = Variable(torch.from_numpy(input_x).cuda(), requires_grad=True)
        else:
            x = Variable(torch.from_numpy(input_x), requires_grad=True)

        # Encoder
        hl_rep = self.activation_function(self.ih_matrix(x))

        # Decoder
        y_out = self.activation_function(self.ho_matrix(hl_rep))

        return y_out, x


class DNN(nn.Module):

    def __init__(self, N, l_dim):
        """
        Constructing blocks for a deep neural network
        for MSS.
        Args :
            N      : (int) Original dimensionallity of the input.
            l_dim  : (int) Dimensionallity of the latent variables.
        """
        super(DNN, self).__init__()
        print('Constructing a Deep Neural Network')
        self._N = N
        self._ldim = l_dim

        self.activation_function = torch.nn.ReLU()

        # Layers
        self.ih_matrix = nn.Linear(self._N, self._ldim, bias=True)
        self.hh_matrix = nn.Linear(self._ldim, self._ldim, bias=True)
        self.hh_b_matrix = nn.Linear(self._ldim, self._ldim, bias=True)
        self.ho_matrix = nn.Linear(self._ldim, self._N, bias=True)

        # Initialize the weights
        self.initialize_ffn()

    def initialize_ffn(self):
        """
            Manual weight/bias initialization.
        """
        # Matrices
        nn.init.xavier_normal(self.ih_matrix.weight)
        nn.init.xavier_normal(self.hh_matrix.weight)
        nn.init.xavier_normal(self.hh_b_matrix.weight)
        nn.init.xavier_normal(self.ho_matrix.weight)

        print('Initialization of the DNN done...')

        return None

    def forward(self, input_x):

        if torch.has_cudnn:
            x = Variable(torch.from_numpy(input_x).cuda(), requires_grad=True)
        else:
            x = Variable(torch.from_numpy(input_x), requires_grad=True)

        hl_rep = self.activation_function(self.ih_matrix(x))
        hl_rep = self.activation_function(self.hh_matrix(hl_rep))
        hl_rep = self.activation_function(self.hh_b_matrix(hl_rep))
        y_out = self.activation_function(self.ho_matrix(hl_rep))

        return y_out, x


class FFN(nn.Module):

    def __init__(self, N):
        """
        Constructing blocks for a single layer FFN,
        for pre-training.
        Args :
            N      : (int) Original dimensionallity of the input.
        """
        super(FFN, self).__init__()
        print('Constructing FFN')
        self._N = N
        self.activation_function = torch.nn.ReLU()

        # Single Layer
        self.io_matrix = nn.Linear(self._N, self._N, bias=True)

        # Initialize the weights
        self.initialize_ffn()

    def initialize_ffn(self):
        """
            Manual weight/bias initialization.
        """
        # Matrix
        nn.init.xavier_normal(self.io_matrix.weight)

        print('Initialization of the FFN done...')

        return None

    def forward(self, input_x):

        if torch.has_cudnn:
            x = Variable(torch.from_numpy(input_x).cuda(), requires_grad=True)
        else:
            x = Variable(torch.from_numpy(input_x), requires_grad=True)

        return self.activation_function(self.io_matrix(x)), x

# EOF
