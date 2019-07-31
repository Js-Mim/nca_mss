# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

import torch
from torch.autograd import Variable
import numpy as np


dtype = torch.DoubleTensor
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
layers = []
biases = []
w_e = Variable(torch.randn(N, H).type(dtype), requires_grad=True)
b_e = Variable(torch.randn(H,).type(dtype), requires_grad=True)

w_d = Variable(torch.randn(H, N_out).type(dtype), requires_grad=True)
b_d = Variable(torch.randn(N_out,).type(dtype), requires_grad=True)

layers.append(w_e)
layers.append(w_d)
biases.append(b_e)
biases.append(b_d)

# Matrices we need the gradients wrt
parameters = torch.nn.ParameterList()
p_e = torch.nn.Parameter(torch.randn(N, H).type(dtype), requires_grad=True)
p_d = torch.nn.Parameter(torch.randn(H, N_out).type(dtype), requires_grad=True)
parameters.append(p_e)
parameters.append(p_d)


# Non-linearity
relu = torch.nn.ReLU()

comb_matrix = torch.autograd.Variable(torch.eye(N), requires_grad=True).double()
for index in range(2):

    b_sc_m = relu(parameters[index].mm((layers[index] + biases[index]).t()))
    b_scaled = layers[index] * b_sc_m

    comb_matrix = torch.matmul(b_scaled, comb_matrix)

y_pred = torch.matmul(comb_matrix, x)
loss = (y - y_pred).norm(1)

loss.backward()
delta_term = (torch.sign(y_pred - y)).mm(x.t())


# With relu
w_tilde_d = relu(parameters[1].mm((layers[1] + biases[1]).t())) * w_d
w_tilde_e = w_e * relu(parameters[0].mm((layers[0] + biases[0]).t()))

relu_grad_dec = p_d.mm((w_d + b_d).t()).gt(0).double()
relu_grad_enc = p_e.mm((w_e + b_e).t()).gt(0).double()

p_d_grad_hat = (delta_term.mm(w_tilde_e.t()) * w_d * relu_grad_dec).mm((w_d + b_d))
p_e_grad_hat = (w_tilde_d.t().mm(delta_term) * w_e * relu_grad_enc).mm((w_e + b_e))

print('Error between autograd computation and calculated:'+str((parameters[1].grad - p_d_grad_hat).abs().max()))
print('Error between autograd computation and calculated:'+str((parameters[0].grad - p_e_grad_hat).abs().max()))

# EOF
