# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from tt_lib.utils import _nearest_y
import math
import numpy as np


def get_sliding_window_op_output_nhw_shape(
    input_n, input_h, input_w, stride_h, stride_w, pad_h, pad_w, window_h, window_w
):
    output_h = ((int)((input_h + (2 * pad_h) - window_h) / stride_h)) + 1
    output_w = ((int)((input_w + (2 * pad_w) - window_w) / stride_w)) + 1
    return [input_n, output_h, output_w]


def get_sliding_window_op_output_shard_nhw_size(
    num_cores_nhw, input_n, input_h, input_w, stride_h, stride_w, pad_h, pad_w, window_h, window_w
):
    output_nhw_shape = get_sliding_window_op_output_nhw_shape(
        input_n, input_h, input_w, stride_h, stride_w, pad_h, pad_w, window_h, window_w
    )
    output_nhw_size_to_shard_evenly = _nearest_y(np.prod(output_nhw_shape), num_cores_nhw * 32)
    output_shard_nhw_size = (int)(output_nhw_size_to_shard_evenly / num_cores_nhw)
    return output_shard_nhw_size


def get_sliding_window_op_output_shard_nhw_size_rm(
    num_cores_nhw, input_n, input_h, input_w, stride_h, stride_w, pad_h, pad_w, window_h, window_w
):
    output_nhw_shape = get_sliding_window_op_output_nhw_shape(
        input_n, input_h, input_w, stride_h, stride_w, pad_h, pad_w, window_h, window_w
    )
    print(f"ncores_nhw: {num_cores_nhw}")
    print(f"output shape: {output_nhw_shape}")
    output_nhw_size_to_shard_evenly = _nearest_y(np.prod(output_nhw_shape), 32)
    print(f"output_nhw_size_to_shard_evenly_new: {output_nhw_size_to_shard_evenly}")
    output_shard_nhw_size = (int)(math.ceil(output_nhw_size_to_shard_evenly / num_cores_nhw))
    print(f"output_shard_nhw_size_new: {output_shard_nhw_size}")
    return output_shard_nhw_size
