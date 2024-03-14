# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import sys

import tt_lib as ttl
import tt_lib.fallback_ops
import tt_lib.profiler as profiler

from loguru import logger


def unet_reshard(
    ttnn_tensor,
    sharded_memory_config,
    use_reshard=True,
    interleaved_memory_config=ttnn.L1_MEMORY_CONFIG,
    tilize=False,
    dtype=None,
):
    if ttnn_tensor.memory_config() == sharded_memory_config:
        return ttnn_tensor

    if use_reshard:
        return ttnn.to_memory_config(
            ttnn_tensor,
            memory_config=sharded_memory_config,
        )
    else:
        ttl_tensor = ttnn_tensor
        if ttl_tensor.is_sharded():
            i = ttl_tensor
            ttl_tensor = ttl.tensor.sharded_to_interleaved(ttl_tensor, interleaved_memory_config)
            ttnn.deallocate(i)
        i = ttl_tensor
        ttl_tensor = ttl.tensor.interleaved_to_sharded(
            ttl_tensor,
            sharded_memory_config,
            None if tilize else dtype,
        )
        ttnn.deallocate(i)
        if tilize:
            i = ttl_tensor
            h, w = list(ttnn_tensor.shape)[2:]
            pad_h = (ttnn.TILE_SIZE - h % ttnn.TILE_SIZE) % ttnn.TILE_SIZE
            pad_w = (ttnn.TILE_SIZE - w % ttnn.TILE_SIZE) % ttnn.TILE_SIZE
            ttl_tensor = ttl.tensor.tilize_with_val_padding(
                ttl_tensor,
                list(ttnn_tensor.shape)[:2] + [h + pad_h, w + pad_w],
                [0, 0, 0, 0],
                0,
                output_mem_config=ttl_tensor.memory_config(),
                output_dtype=dtype,
            )
            ttnn.deallocate(i)
        return ttl_tensor


def unet_concat(ttnn_tensors, dim=-1, use_reshard=True):
    assert len(ttnn_tensors) > 0
    assert dim < 0
    rank = len(ttnn_tensors[0].shape)
    ttlib_tensors = ttnn_tensors
    all_sharded = all(t.is_sharded() for t in ttlib_tensors)
    if all_sharded:
        max_idx, output_mem_config = max(
            ((i, t.memory_config()) for i, t in enumerate(ttlib_tensors)), key=lambda m: m[1].shard_spec.num_cores()
        )
        for i in range(0, len(ttlib_tensors)):
            if i == max_idx:
                continue
            t = ttlib_tensors[i]
            t_mem_config = t.memory_config()
            t_shard_shape = t_mem_config.shard_spec.shape
            output_shard_shape = output_mem_config.shard_spec.shape
            output_shard_shape[dim] += t_shard_shape[dim]
            output_mem_config.shard_spec.shape = output_shard_shape

            reshard_shape = output_shard_shape
            reshard_shape[dim] = t_shard_shape[dim]
            if reshard_shape != t_shard_shape:
                t_mem_config.shard_spec.shape = reshard_shape
                t_mem_config.shard_spec.grid = output_mem_config.shard_spec.grid
                t_mem_config.shard_spec.orientation = output_mem_config.shard_spec.orientation
                ttlib_tensors[i] = unet_reshard(t, t_mem_config, use_reshard=use_reshard)
    else:
        output_mem_config = ttlib_tensors[0].memory_config()
        for i in range(len(ttlib_tensors)):
            if ttlib_tensors[i].is_sharded():
                ttlib_tensors[i] = ttl.tensor.sharded_to_interleaved(ttlib_tensors[i], ttnn.L1_MEMORY_CONFIG)

    dim = dim + 4 - rank
    return ttl.tensor.concat(ttlib_tensors, dim=dim, output_mem_config=output_mem_config)


class UNet:
    def __init__(
        self,
        parameters,
    ) -> None:
        self.c1 = parameters.c1
        self.c1_2 = parameters.c1_2
        self.p1 = parameters.p1
        self.c2 = parameters.c2
        self.c2_2 = parameters.c2_2
        self.p2 = parameters.p2
        self.c3 = parameters.c3
        self.c3_2 = parameters.c3_2
        self.p3 = parameters.p3
        self.c4 = parameters.c4
        self.c4_2 = parameters.c4_2
        self.p4 = parameters.p4
        self.bnc = parameters.bnc
        self.bnc_2 = parameters.bnc_2
        self.c5 = parameters.c5
        self.c5_2 = parameters.c5_2
        self.c5_3 = parameters.c5_3
        self.c6 = parameters.c6
        self.c6_2 = parameters.c6_2
        self.c6_3 = parameters.c6_3
        self.c7 = parameters.c7
        self.c7_2 = parameters.c7_2
        self.c7_3 = parameters.c7_3
        self.c8 = parameters.c8
        self.c8_2 = parameters.c8_2
        self.c8_3 = parameters.c8_3
        self.output_layer = parameters.output_layer

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.c1.conv.input_sharded_memory_config)

        profiler.signpost("c1")
        output_tensor = self.c1(input_tensor)
        output_tensor = self.c1_2(output_tensor)
        save_c1_2_out = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = self.p1(output_tensor)

        profiler.signpost("c2")
        output_tensor = unet_reshard(output_tensor, self.c2.conv.input_sharded_memory_config)
        output_tensor = self.c2(output_tensor)
        output_tensor = self.c2_2(output_tensor)
        save_c2_2_out = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = self.p2(output_tensor)

        profiler.signpost("c3")
        output_tensor = unet_reshard(output_tensor, self.c3.conv.input_sharded_memory_config)
        output_tensor = self.c3(output_tensor)
        output_tensor = self.c3_2(output_tensor)
        save_c3_2_out = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = self.p3(output_tensor)

        profiler.signpost("c4")
        output_tensor = unet_reshard(output_tensor, self.c4.conv.input_sharded_memory_config)
        output_tensor = self.c4(output_tensor)
        output_tensor = self.c4_2(output_tensor)
        save_c4_2_out = output_tensor
        output_tensor = self.p4(output_tensor)

        profiler.signpost("bnc")
        output_tensor = self.bnc(output_tensor)
        output_tensor = self.bnc_2(output_tensor)

        ## upsample block
        profiler.signpost("upsample1")
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.reshape(output_tensor, (1, 132, 10, 64))
        output_tensor = ttnn.upsample(output_tensor, 2)
        output_tensor = ttnn.reshape(output_tensor, (1, 1, 5280, 64))

        profiler.signpost("concat1")
        output_tensor = unet_concat([output_tensor, save_c4_2_out], dim=-1)

        profiler.signpost("c5")
        output_tensor = unet_reshard(output_tensor, self.c5.conv.input_sharded_memory_config)
        output_tensor = self.c5(output_tensor)
        output_tensor = self.c5_2(output_tensor)
        output_tensor = self.c5_3(output_tensor)

        profiler.signpost("upsample2")
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.reshape(output_tensor, (1, 264, 20, 32))
        output_tensor = ttnn.upsample(output_tensor, 2)
        output_tensor = ttnn.reshape(output_tensor, (1, 1, 21120, 32))

        profiler.signpost("concat2")
        output_tensor = unet_concat([output_tensor, save_c3_2_out], dim=-1)

        profiler.signpost("c6")
        output_tensor = unet_reshard(output_tensor, self.c6.conv.input_sharded_memory_config)
        output_tensor = self.c6(output_tensor)
        output_tensor = self.c6_2(output_tensor)
        output_tensor = self.c6_3(output_tensor)

        profiler.signpost("upsample3")
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.reshape(output_tensor, (1, 528, 40, 32))
        output_tensor = ttnn.upsample(output_tensor, 2)
        output_tensor = ttnn.reshape(output_tensor, (1, 1, 84480, 32))

        profiler.signpost("concat3")
        output_tensor = unet_concat([output_tensor, save_c2_2_out], dim=-1)

        profiler.signpost("c7")
        hacked_shard_shape = self.c7.conv.input_sharded_memory_config.shard_spec.shape
        hacked_shard_shape[1] = output_tensor.shape[-1]
        self.c7.conv.input_sharded_memory_config.shard_spec.shape = hacked_shard_shape
        output_tensor = self.c7(output_tensor)
        output_tensor = self.c7_2(output_tensor)
        output_tensor = self.c7_3(output_tensor)

        profiler.signpost("upsample4")
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.reshape(output_tensor, (1, 1056, 80, 16))
        output_tensor = ttnn.upsample(output_tensor, 2)
        output_tensor = ttnn.reshape(output_tensor, (1, 1, 160 * 1056 * 2, 16))

        profiler.signpost("concat4")
        output_tensor = unet_concat([output_tensor, save_c1_2_out], dim=-1)

        profiler.signpost("c8")
        output_tensor = self.c8(output_tensor)
        output_tensor = self.c8_2(output_tensor)
        output_tensor = self.c8_3(output_tensor)
        output_tensor = self.output_layer(output_tensor)
        return ttnn.from_device(output_tensor)
