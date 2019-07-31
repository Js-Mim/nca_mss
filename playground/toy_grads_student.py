# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

import torch
from torch.autograd import Variable
import numpy as np


dtype = torch.FloatTensor
np.random.seed(2183)
torch.manual_seed(2183)


# D is the "batch size"; N is input dimension;
# H is hidden dimension; N_out is output dimension.
D, N, H, N_out = 1, 20, 20, 20

# Create random Tensors to hold input and outputs, and wrap them in Variables.
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these Variables during the backward pass.
x = Variable(torch.randn(N, D).type(dtype), requires_grad=True)
y = Variable(torch.randn(N_out, D).type(dtype), requires_grad=False)

# Create random Tensors for weights, and wrap them in Variables.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Variables during the backward pass.
c = Variable(torch.randn(N, H).type(dtype), requires_grad=True)

y_pred = c.mm(x)
loss = (y - y_pred).norm(1)
loss.backward()


delta = torch.sign(y_pred - y).mm(x.t())
print((c.grad - delta).abs().max())

"""
# Valid for 1D case
dEdy = torch.autograd.grad(loss, y_pred,
                           grad_outputs=torch.ones(y_pred.size()),
                           retain_graph=True)[0].data.numpy()

dydC = torch.autograd.grad(y_pred, c,
                           grad_outputs=torch.ones(c.size()),
                           retain_graph=True)[0].data.numpy()
"""

# EOF
