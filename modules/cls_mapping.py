# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class MappingEstimator(nn.Module):

    def __init__(self, pytorch_model, keyword='matrix', optimization=True):
        super(MappingEstimator, self).__init__()
        print('Constructing Mapping Estimator')
        # Module initialization
        self.sizes = []
        self.use_bias = []
        self.sc_matrices = nn.ModuleList()
        self.old_weights = []
        self.old_biases = []
        self.activation_function = torch.nn.ReLU()
        self.opt = optimization

        # Inspecting given model
        model_inspector = pytorch_model.named_modules()

        for name, submodule in model_inspector:
            if keyword in name:
                self.sizes += [submodule.weight.size(1)]
                self.sc_matrices.append(nn.Linear(self.sizes[-1], self.sizes[-1], bias=False))
                self.old_weights.append(Variable(submodule.weight.data, requires_grad=True))
                try:
                    self.old_biases.append(Variable(submodule.bias.data, requires_grad=True))
                    self.use_bias += [True]
                except AttributeError:
                    self.use_bias += [False]

        print('Initializing parameters')
        self.initialize_map()

    def initialize_map(self):
        for item_index in range(len(self.sc_matrices)):
            if self.opt:
                torch.nn.init.xavier_normal(self.sc_matrices[item_index].weight)
            else:
                self.sc_matrices[item_index].weight.data.fill_(1.)

        print('Initialization done...')

        return None

    def forward(self, xs):
        # Initialize with an idenity matrix
        comb_matrix = Variable(torch.eye(self.sizes[0]), requires_grad=True)

        for item_index in range(len(self.sc_matrices)):
            if self.use_bias[item_index]:
                b_sc_vec = self.sc_matrices[item_index](self.old_weights[item_index] + self.old_biases[item_index])
            else:
                b_sc_vec = self.sc_matrices[item_index](self.old_weights[item_index])

            b_scaled = self.activation_function(b_sc_vec) * self.old_weights[item_index]
            comb_matrix = torch.matmul(b_scaled, comb_matrix)

        xt_hat = torch.matmul(comb_matrix, xs.t()).t()

        self.mapping = comb_matrix

        return xt_hat

    def map(self, xs):
        return torch.matmul(self.mapping, xs.t()).t()

    def learn_mapping(self, xs, xt, epochs=600, lr=1e-4, verbose=False):

        print('Number of matrices: ' + str(len(list(self.sc_matrices.parameters()))))
        # Optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(list(self.sc_matrices.parameters()), lr=lr)

        loss_data = []
        for epoch in range(epochs):
            xs_cuda = Variable(torch.from_numpy(xs).cuda(), requires_grad=True)
            xt_cuda = Variable(torch.from_numpy(xt).cuda(), requires_grad=True)

            xt_hat = self.forward(xs_cuda)
            loss = torch.norm(xt_cuda-xt_hat, p=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_data += [loss.data[0]]

            if verbose:
                print(np.mean(loss_data))

        print('Terminated with L1 Loss: ' + str(np.mean(loss_data)))
        return None


class MappingEstimatorStudent(nn.Module):

    def __init__(self, N):
        super(MappingEstimatorStudent, self).__init__()
        print('Constructing Student Mapping Estimator')
        # Module initialization
        self.student = nn.Linear(N, N, bias=False)

        print('Initializing parameters')
        self.initialize_student()

    def initialize_student(self):
        torch.nn.init.xavier_normal(self.student.weight)
        print('Initialization done...')

        return None

    def forward(self, xs):
        self.mapping = self.student.weight
        return self.student(xs)

    def learn_mapping(self, xs, xt, epochs=600, lr=1e-4, verbose=False):

        # Optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(list(self.student.parameters()), lr=lr)

        loss_data = []
        for epoch in range(epochs):
            xs_cuda = Variable(torch.from_numpy(xs).cuda(), requires_grad=True)
            xt_cuda = Variable(torch.from_numpy(xt).cuda(), requires_grad=True)

            xt_hat = self.forward(xs_cuda)
            loss = torch.norm(xt_cuda - xt_hat, p=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_data += [loss.data[0]]

            if verbose:
                print(np.mean(loss_data))

        print('Terminated with L1 Loss: ' + str(np.mean(loss_data)))
        return None

# EOF
