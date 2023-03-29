import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/")
sys.path.append("/home/farbabi/git/tt-metal")
print(f, sys.path)
import torch
from torch import nn
from torchvision import transforms, datasets

import pymetal
from pymetal import ttmetal as ttm
from utility_functions import tilize_to_list, untilize
from batch_norm import batchnorm1d_inference
from sweep_tests.comparison_funcs import comp_pcc

epsilon = 1e-5

def ttLinear(weight, bias):

    def linear_(activation):
        weight_T = ttm.tensor.transpose(weight)
        output = ttm.tensor.matmul(activation, weight_T)
        output_plus_bias = ttm.tensor.bcast(output, bias, ttm.tensor.BcastOpMath.ADD, ttm.tensor.BcastOpDim.H)
        return output_plus_bias

    return linear_


def torchLinear(in_features, out_features, weight, bias):
    linear_torch = torch.nn.Linear(out_features, in_features)
    linear_torch.weight = nn.Parameter(weight)
    linear_torch.bias = nn.Parameter(bias)

    return linear_torch


def ttBatchnorm1d_inference(gamma, beta, running_mean, running_var, epsilon):

    BCHW = ttm.tensor.BcastOpDim.HW
    BCADD = ttm.tensor.BcastOpMath.ADD

    def batchnorm1d_inference_(X):
        var_plus_eps = ttm.tensor.bcast(running_var, epsilon, BCADD, BCHW)
        sqrt_var = ttm.tensor.sqrt(var_plus_eps)
        sqrt_inv = ttm.tensor.recip(sqrt_var)
        x_minus_mean = ttm.tensor.sub(X, running_mean)
        x_div_sqrt = ttm.tensor.mul(x_minus_mean, sqrt_inv)
        x_gamma = ttm.tensor.mul(x_div_sqrt, gamma)
        Y = ttm.tensor.add(x_gamma, beta)
        return Y

    return batchnorm1d_inference_


class PytorchBatchNorm1D(nn.Module):
    def __init__(self, input_dim):
        super(PytorchBatchNorm1D, self).__init__()

        self.batchnorm1d_1 = nn.BatchNorm1d(input_dim)

    def forward(self, x):

        bn1_out =  self.batchnorm1d_1(x)

        return bn1_out


def run_block_inference(in_features, out_features):

    # set inputs
    inputs_torch = torch.FloatTensor(1, in_features).uniform_(-1., 1.).requires_grad_(True)

    inputs_reshape = inputs_torch.reshape(1, 1, 1, -1)
    inputs_targ = torch.zeros(1, 1, 32, inputs_reshape.shape[3])
    inputs_targ[:, :, :1, :] = inputs_reshape
    tilized_inputs = tilize_to_list(inputs_targ)
    inputs_tt = ttm.tensor.Tensor(tilized_inputs, inputs_targ.shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    # torch linear params
    weight_lin_torch = torch.randn(out_features, in_features)
    bias_lin_torch = torch.randn(out_features)
    linear_torch = torchLinear(in_features, out_features, weight_lin_torch, bias_lin_torch)

    # tt linear params
    weight_lin = weight_lin_torch.view(1, 1, out_features, in_features)
    tilized_weight_lin_tt = tilize_to_list(weight_lin)
    weight_lin_tt = ttm.tensor.Tensor(tilized_weight_lin_tt, weight_lin.shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    bias_lin_src = bias_lin_torch.view(1, 1, 1, out_features)
    bias_lin = torch.zeros(1, 1, 32, out_features)
    bias_lin[:, :, :1, :] = bias_lin_src
    tilized_bias_lin_tt = tilize_to_list(bias_lin)
    bias_lin_tt = ttm.tensor.Tensor(tilized_bias_lin_tt, bias_lin.shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    # batch norm torch
    bn_torch = PytorchBatchNorm1D(out_features)
    bn_torch.eval()
    weight_bn_torch = torch.nn.Parameter(torch.FloatTensor(out_features).uniform_(-1., 1.).requires_grad_(True))
    bias_bn_torch =  torch.nn.Parameter(torch.FloatTensor(out_features).uniform_(-1., 1.).requires_grad_(True))
    running_mean_bn_torch = torch.FloatTensor(out_features).uniform_(-1., 1.).requires_grad_(False)
    running_var_bn_torch = torch.FloatTensor(out_features).uniform_(0., 1.).requires_grad_(False)  #must be positive

    bn_torch.batchnorm1d_1.weight = weight_bn_torch
    bn_torch.batchnorm1d_1.bias = bias_bn_torch
    bn_torch.batchnorm1d_1.running_mean = running_mean_bn_torch
    bn_torch.batchnorm1d_1.running_var = running_var_bn_torch
    bn_torch.batchnorm1d_1.eps = epsilon

    # batch norm tt
    weight_bn_src = weight_bn_torch.view(1, 1, 1, out_features)
    weight_bn_tt = torch.zeros(1, 1, 32, out_features)
    weight_bn_tt[:, :, :1, :] = weight_bn_src
    tilized_weight_bn_tt= tilize_to_list(weight_bn_tt)
    gamma = ttm.tensor.Tensor(tilized_weight_bn_tt, [1, 1, 32, out_features], ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    bias_bn_src = bias_bn_torch.view(1, 1, 1, out_features)
    bias_bn_tt = torch.zeros(1, 1, 32, out_features)
    bias_bn_tt[:, :, :1, :] = bias_bn_src
    tilized_bias_bn_tt= tilize_to_list(bias_bn_tt)
    beta = ttm.tensor.Tensor(tilized_bias_bn_tt, [1, 1, 32, out_features], ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    running_mean_bn_src = running_mean_bn_torch.view(1, 1, 1, out_features)
    running_mean_bn_tt = torch.zeros(1, 1, 32, out_features)
    running_mean_bn_tt[:, :, :1, :] = running_mean_bn_src
    tilized_running_mean_tt= tilize_to_list(running_mean_bn_tt)
    running_mean_tt = ttm.tensor.Tensor(tilized_running_mean_tt, [1, 1, 32, out_features], ttm.tensor.DataType.BFLOAT16,ttm.tensor.Layout.TILE, device)

    running_var_bn_src = running_var_bn_torch.view(1, 1, 1, out_features)
    running_var_bn_tt = torch.zeros(1, 1, 32, out_features)
    running_var_bn_tt[:, :, :1, :] = running_var_bn_src
    tilized_running_var_tt = tilize_to_list(running_var_bn_tt)
    running_var_tt = ttm.tensor.Tensor(tilized_running_var_tt, [1, 1, 32, out_features], ttm.tensor.DataType.BFLOAT16,ttm.tensor.Layout.TILE, device)

    epsilon_tt = ttm.tensor.Tensor([epsilon] + [0 for _ in range(32 * 32 - 1)], [1, 1, 32, 32], ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    # run through the models
    # torch
    output_lin_torch = linear_torch(inputs_torch)
    output_bn_torch = bn_torch(output_lin_torch)
    output_full_torch = torch.nn.functional.relu(output_bn_torch)

    # tt
    linear_tt = ttLinear(weight_lin_tt, bias_lin_tt)
    output_lin_tt = linear_tt(inputs_tt)
    bn_tt =  ttBatchnorm1d_inference(gamma, beta, running_mean_tt, running_var_tt, epsilon_tt)
    output_bn_tt = bn_tt(output_lin_tt)
    output_full_tt = ttm.tensor.relu(output_bn_tt)

    # compare
    output_lin_tt_untilized = untilize(torch.Tensor(output_lin_tt.to(host).data()).reshape(output_lin_tt.shape()))
    output_lin_tt_untilized = output_lin_tt_untilized[0, 0, 0, :]

    output_bn_tt_untilized = untilize(torch.Tensor(output_bn_tt.to(host).data()).reshape(output_bn_tt.shape()))
    output_bn_tt_untilized = output_bn_tt_untilized[0, 0, 0, :]

    output_full_tt_untilized = untilize(torch.Tensor(output_full_tt.to(host).data()).reshape(output_full_tt.shape()))
    output_full_tt_untilized = output_full_tt_untilized[0, 0, 0, :]

    print('pytorch_linear_out:', output_lin_torch[0][0:10])
    print('tt_linear_out:', output_lin_tt_untilized[0:10])

    liner_test_result = comp_pcc(output_lin_torch[0], output_lin_tt_untilized)
    print('\n\n', liner_test_result, '\n\n')

    print('pytorch_bn_out:', output_bn_torch[0][0:10])
    print('tt_bn_out:', output_bn_tt_untilized[0:10])

    bn_test_result = comp_pcc(output_bn_torch[0], output_bn_tt_untilized)
    print('\n\n', bn_test_result, '\n\n')

    print('pytorch_full_out:', output_full_torch[0][0:10])
    print('tt_full_out:', output_full_tt_untilized[0:10])

    full_test_result = comp_pcc(output_full_torch[0], output_full_tt_untilized)
    print('\n\n', full_test_result, '\n\n')

if __name__ == "__main__":
    # Initialize the device
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()
    run_block_inference(1024, 256)
    ttm.device.CloseDevice(device)
