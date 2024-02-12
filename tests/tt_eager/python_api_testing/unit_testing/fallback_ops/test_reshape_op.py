# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib as ttl
import tt_lib.fallback_ops
from models.utility_functions import (
    comp_allclose_and_pcc,
    comp_pcc,
)
from loguru import logger
import pytest


@pytest.mark.parametrize(
    "input_shape, output_shape, on_device",
    (
        (torch.Size([1, 3, 6, 4]), torch.Size([3, 1, 3, 8]), True),
        (torch.Size([1, 3, 6, 4]), torch.Size([3, 1, 3, 8]), False),
        (torch.Size([1, 2, 64, 32]), torch.Size([2, 4, 128, 4]), True),
        (torch.Size([1, 2, 64, 32]), torch.Size([2, 4, 128, 4]), False),
    ),
)
def test_reshape_fallback(input_shape, output_shape, on_device, device):
    torch.manual_seed(1234)

    x = torch.randn(input_shape).bfloat16().float()
    pt_out = torch.reshape(x, output_shape)

    # Test on host RM
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.fallback_ops.reshape(t0, *output_shape)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    comp_pass, _ = comp_pcc(pt_out, output, 0.9999)
    _, comp_out = comp_allclose_and_pcc(pt_out, output)
    logger.info(comp_out)
    assert comp_pass
