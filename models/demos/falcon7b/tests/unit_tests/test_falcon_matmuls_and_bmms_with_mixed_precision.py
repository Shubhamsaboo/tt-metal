# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

import tt_lib as ttl
from models.utility_functions import comp_pcc, tt2torch_tensor, skip_for_wormhole_b0
import torch


def run_falcon_matmul_test(
    falcon_op,
    seq_len,
    in0_dtype,
    in1_dtype,
    out_dtype,
    in0_mem_config,
    in1_mem_config,
    out_mem_config,
    device,
):
    pcc = 0.99
    if out_dtype == ttl.tensor.DataType.BFLOAT8_B:
        pcc = 0.98

    if falcon_op == ttl.tensor.falcon_fused_qkv_matmul:
        a_shape = [1, 1, seq_len, 4544]
        b_shape = [1, 1, 4544, 4672]
        expected_output_shape = [1, 1, seq_len, 4672]
    elif falcon_op == ttl.tensor.falcon_selfout_matmul:
        a_shape = [1, 1, seq_len, 4544]
        b_shape = [1, 1, 4544, 4544]
        expected_output_shape = [1, 1, seq_len, 4544]
    elif falcon_op == ttl.tensor.falcon_dense_4h_to_h_matmul:
        a_shape = [1, 1, seq_len, 18176]
        b_shape = [1, 1, 18176, 4544]
        expected_output_shape = [1, 1, seq_len, 4544]

        if (seq_len == 1024 and in0_dtype == in1_dtype == out_dtype == ttl.tensor.DataType.BFLOAT16) or (
            seq_len == 2048
            and (
                in0_dtype == ttl.tensor.DataType.BFLOAT16
                or in1_dtype == ttl.tensor.DataType.BFLOAT16
                or out_dtype == ttl.tensor.DataType.BFLOAT16
            )
        ):
            logger.warning(
                f"For seq_len: {seq_len}, in0_dtype: {in0_dtype}, in1_dtype: {in1_dtype}, and out_dtype: {out_dtype}, L1 space is not enough. Running with in0, in1, and out on DRAM instead!"
            )
            in0_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
            )
            in1_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
            )
            out_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
            )
    elif falcon_op == ttl.tensor.falcon_dense_h_to_4h_matmul:
        a_shape = [1, 1, seq_len, 4544]
        b_shape = [1, 1, 4544, 18176]
        expected_output_shape = [1, 1, seq_len, 18176]

        in0_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
        out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)

        if seq_len == 2048 and out_dtype == ttl.tensor.DataType.BFLOAT16:
            logger.warning(
                f"For seq_len: {seq_len}, in0_dtype: {in0_dtype}, in1_dtype: {in1_dtype}, and out_dtype: {out_dtype}, L1 space is not enough. Running with in0, in1, and out on DRAM instead!"
            )
            in0_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
            )
            in1_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
            )
            out_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
            )
    elif falcon_op == ttl.tensor.falcon_lm_head_matmul:
        a_shape = [1, 1, seq_len, 4544]
        b_shape = [1, 1, 4544, 65024]
        expected_output_shape = [1, 1, seq_len, 65024]

        if (
            seq_len == 512
            and (
                in0_dtype == ttl.tensor.DataType.BFLOAT16
                or in1_dtype == ttl.tensor.DataType.BFLOAT16
                or out_dtype == ttl.tensor.DataType.BFLOAT16
            )
            or seq_len == 1024
            or seq_len == 2048
        ):
            logger.warning(
                f"For seq_len: {seq_len}, in0_dtype: {in0_dtype}, in1_dtype: {in1_dtype}, and out_dtype: {out_dtype}, L1 space is not enough. Running with in0, in1, and out on DRAM instead!"
            )
            in0_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
            )
            in1_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
            )
            out_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
            )
    elif falcon_op == ttl.tensor.matmul:
        # attn matmul, post-transpose
        a_shape = [1, 71, seq_len, 64]
        b_shape = [1, 1, 64, seq_len]
        expected_output_shape = [1, 71, seq_len, seq_len]

        in0_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
        out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)

    else:
        raise NotImplementedError(f"falcon matmul op is undefined!")

    torch.manual_seed(1234)

    A = torch.randn(a_shape)
    B = torch.randn(b_shape) - 0.95

    a_t = ttl.tensor.Tensor(A, in0_dtype).to(ttl.tensor.Layout.TILE).to(device, in0_mem_config)
    b_t = ttl.tensor.Tensor(B, in1_dtype).to(ttl.tensor.Layout.TILE).to(device, in1_mem_config)
    bias_t = None

    if falcon_op == ttl.tensor.falcon_dense_4h_to_h_matmul:
        compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # A = [32, 568] * [568, 142]
        program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            in0_block_w=8,
            per_core_M=4,
            per_core_N=18,
            out_subblock_h=1,
            out_subblock_w=6,
            transpose_mcast=False,
            fused_activation=None,
        )

        out = ttl.operations.primary.matmul(
            a_t,
            b_t,
            program_config=program_config,
            output_mem_config=out_mem_config,
            output_dtype=out_dtype,
            compute_kernel_config=compute_kernel_config,
        )

        # out = falcon_op(a_t, b_t, bias_t, output_mem_config=out_mem_config, output_dtype=out_dtype)
    elif falcon_op == ttl.tensor.falcon_dense_h_to_4h_matmul:
        compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # # A = [32, 142] B = [142, 568]
        # # Split on 8 cores for x -> Am = [4, 142], per_core_M = 4
        # # Split on 8 cores for y -> Bn = [142, 71], per_core_N = 71
        # program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        #     compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        #     in0_block_w=2,
        #     per_core_M=4,
        #     per_core_N=71,
        #     out_subblock_h=4,
        #     out_subblock_w=1,
        #     transpose_mcast=False,
        #     fused_activation=None,
        # )

        # best config ~ 2.6 mil cycles
        program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            in0_block_w=2,
            per_core_M=32,
            per_core_N=9,
            out_subblock_h=2,
            out_subblock_w=3,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        # This is not working
        # Theoretically:
        # A = [32, 142] B = [142, 568]
        # per_core_M = 16 => num_x_cores=2, per_core_A = [16, 142]
        # per_core_N = 18 => num_y_cores=32, per_core_B = [142, 18]
        # Total num_cores = num_x_cores * num_y_cores = 64
        # 1D doesn't assert with this, but PCC is bad
        # 2D asserts as it has constraints such that max_num_x_cores = 8 and max_num_y_cores = 8
        # Note: 568 / 18 == 17.66 => pad to 18
        # program_config =ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        #     compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        #     in0_block_w = 2,
        #     per_core_M = 16,
        #     per_core_N= 18,
        #     out_subblock_h = 1,
        #     out_subblock_w = 1,
        #     fuse_batch = True,
        #     fused_activation = None,
        #     mcast_in0 = True
        # )

        out = ttl.operations.primary.matmul(
            a_t,
            b_t,
            program_config=program_config,
            output_mem_config=out_mem_config,
            output_dtype=out_dtype,
            compute_kernel_config=compute_kernel_config,
        )
    elif falcon_op == ttl.tensor.matmul:
        compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.HiFi4,  # try this lower
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        #     compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        #     in0_block_w=8,
        #     per_core_M=4,
        #     per_core_N=18,
        #     out_subblock_h=1,
        #     out_subblock_w=6,
        #     transpose_mcast=False,
        #     fused_activation=None,
        # )
        # Dram bound
        # Dimensions:
        #         a_shape = [1, 71, seq_len, 64]
        #         b_shape = [1, 1, 64, seq_len]
        #         entire batch doesn't fit
        program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            in0_block_w=1,
            per_core_M=4,
            per_core_N=4,
            out_subblock_h=1,
            out_subblock_w=1,
            transpose_mcast=False,
            fused_activation=None,
        )

        # program_config =ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        #     compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        #     in0_block_w = 2,
        #     per_core_M = 1,
        #     per_core_N= 32,
        #     out_subblock_h = 1,
        #     out_subblock_w = 1,
        #     fuse_batch = False,
        #     fused_activation = None,
        #     mcast_in0 = False
        # )

        out = ttl.operations.primary.matmul(
            a_t,
            b_t,
            program_config=program_config,
            output_mem_config=out_mem_config,
            output_dtype=out_dtype,
            compute_kernel_config=compute_kernel_config,
        )

        # out = ttl.tensor.matmul(
        #     a_t,
        #     b_t,
        #     output_mem_config = out_mem_config
        # )
    else:
        out = falcon_op(a_t, b_t, bias_t, output_mem_config=out_mem_config, output_dtype=out_dtype)

    # Check memory and dtype of inputs and outputs
    assert a_t.memory_config().buffer_type == in0_mem_config.buffer_type
    assert a_t.dtype() == in0_dtype
    assert b_t.memory_config().buffer_type == in1_mem_config.buffer_type
    assert b_t.dtype() == in1_dtype
    assert out.memory_config().buffer_type == out_mem_config.buffer_type
    assert out.dtype() == out_dtype
    logger.debug(f"in0 ({a_shape}): {a_t.memory_config().buffer_type} and {a_t.dtype()}")
    logger.debug(f"in1 ({b_shape}): {b_t.memory_config().buffer_type} and {b_t.dtype()}")
    logger.debug(f"out ({expected_output_shape}): {out.memory_config().buffer_type} and {out.dtype()}")

    assert out.shape() == expected_output_shape
    pyt_got_back_rm = tt2torch_tensor(out)

    ref_bmm = torch.matmul(A, B)

    passing_pcc, output_pcc = comp_pcc(ref_bmm, pyt_got_back_rm, pcc)
    logger.info(f"Passing={passing_pcc}")
    logger.info(f"Output pcc={output_pcc}")

    assert passing_pcc


@pytest.mark.parametrize(
    "in0_mem_config, in1_mem_config, out_mem_config",
    (
        (
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        ),
    ),
    ids=["weights_DRAM"],
)
@pytest.mark.parametrize(
    "out_dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["out_BFLOAT8_B", "out_BFLOAT16"],
)
@pytest.mark.parametrize(
    "in1_dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["in1_BFLOAT8_B", "in1_BFLOAT16"],
)
@pytest.mark.parametrize(
    "in0_dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["in0_BFLOAT8_B", "in0_BFLOAT16"],
)
@pytest.mark.parametrize(
    "falcon_op",
    (
        ttl.tensor.falcon_fused_qkv_matmul,
        ttl.tensor.falcon_selfout_matmul,
        ttl.tensor.falcon_dense_4h_to_h_matmul,
        ttl.tensor.falcon_dense_h_to_4h_matmul,
        ttl.tensor.falcon_lm_head_matmul,
        ttl.tensor.matmul,
    ),
    ids=["fused_qkv", "selfout", "dense_4h_to_h", "dense_h_to_4h", "lm_head", "post_transpose_mm"],
)
@pytest.mark.parametrize(
    "seq_len",
    (128, 256, 512, 1024, 2048),
    ids=["seq_len_128", "seq_len_256", "seq_len_512", "seq_len_1024", "seq_len_2048"],
)
def test_falcon_matmul(
    falcon_op,
    seq_len,
    in0_dtype,
    in1_dtype,
    out_dtype,
    in0_mem_config,
    in1_mem_config,
    out_mem_config,
    request,
    device,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    is_e75_grid_size = (compute_grid_size.x * compute_grid_size.y) == 88
    if is_e75_grid_size and (seq_len == 512) and (falcon_op == ttl.tensor.falcon_lm_head_matmul):
        pytest.skip(f"LM Head does not work on E75 grid size {compute_grid_size}")

    ttl.profiler.set_profiler_location(f"falcon_{request.node.callspec.id}")
    run_falcon_matmul_test(
        falcon_op,
        seq_len,
        in0_dtype,
        in1_dtype,
        out_dtype,
        in0_mem_config,
        in1_mem_config,
        out_mem_config,
        device,
    )
