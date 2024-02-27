# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import tt_lib
import ttnn
from models.utility_functions import torch2tt_tensor, pad_by_zero, tt2torch_tensor, nearest_32
from models.demos.llama2_70b.tt.llama_attention_optimized import TtLlamaAttention_optimized
from models.demos.llama2_70b.tt.llama_mlp_optimized import TtLlamaMLP_optimized
from models.demos.llama2_70b.tt.llama_common import tt_all_gather_torch
from models.demos.llama2_70b.tt.llama_common import generate_rot_emb, gather_rotary_emb


class TtLlamaDecoder_optimized:
    def __init__(self, devices, state_dict, base_url, layer_num, model_config, configuration, batch, emulated=False):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.num_devices = len(devices)
        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.n_local_heads = self.n_heads // self.num_devices
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = configuration.max_seq_len
        self.model_config = model_config
        self.emulated = emulated

        layer_name = f"{base_url}.{layer_num}"

        attn_norm_str = f"{layer_name}.attention_norm.weight"
        ffn_norm_str = f"{layer_name}.ffn_norm.weight"

        self.norm_eps = configuration.norm_eps

        self.attn_norm_list = []
        self.ffn_norm_list = []
        for i in range(self.num_devices):
            attn_norm = tt_lib.tensor.Tensor(
                # Expand to size of input since we decomped norm
                self.state_dict[attn_norm_str].reshape([1, 1, -1, 32]),
                self.model_config["LN_ATTN_WEIGHTS_DTYPE"],
            ).to(devices[i], self.model_config["LN_ATTN_WEIGHTS_MEMCFG"])

            ffn_norm = tt_lib.tensor.Tensor(
                # Expand to size of input since we decomped norm
                self.state_dict[ffn_norm_str].reshape([1, 1, -1, 32]),
                self.model_config["LN_MLP_WEIGHTS_DTYPE"],
            ).to(devices[i], self.model_config["LN_MLP_WEIGHTS_MEMCFG"])

            self.attn_norm_list.append(attn_norm)
            self.ffn_norm_list.append(ffn_norm)

        self.attention = TtLlamaAttention_optimized(
            devices, state_dict, base_url, layer_num, model_config, configuration, emulated=emulated
        )

        self.mlp = TtLlamaMLP_optimized(
            devices,
            state_dict,
            base_url,
            layer_num,
            self.hidden_size,
            model_config,
            emulated=emulated,
        )
        self.rot_emb = generate_rot_emb(self.head_dim, self.max_seq_len * 2)

    def prepare_inputs(self, x, start_pos):
        # Only called by decoder tests
        assert x.size(2) == self.hidden_size
        assert len(x.size()) == 3

        batch = x.size(0)
        seq_len = x.size(1)
        assert seq_len == 1, "Only supporting decode mode"
        x = x.transpose(0, 1).unsqueeze(1)  # [seq_len, 1, batch, hidden_dim]

        position_ids = torch.ones(seq_len, batch, dtype=torch.long) * start_pos
        rot_mat = gather_rotary_emb(self.rot_emb, position_ids)[:, :1]

        padded_layer_past_len = nearest_32(start_pos + 1)
        attn_mask = torch.zeros(seq_len, 1, batch, padded_layer_past_len)
        attn_mask[:, :, :, start_pos + 1 :] = torch.finfo(attn_mask.dtype).min
        attn_mask = attn_mask.expand(-1, self.n_local_heads, -1, -1)

        # expected shapes:
        # x: (seq_len, 1, batch, hidden_dim)
        # start_pos: int
        # rot_mat: [1, 1, head_dim, head_dim]
        # attn_mask: [seq_len, n_heads, batch, padded_layer_past_len]
        assert x.size() == (seq_len, 1, batch, self.hidden_size)
        assert rot_mat.size() == (1, 1, self.head_dim, self.head_dim)
        assert attn_mask.size() == (seq_len, self.n_local_heads, batch, padded_layer_past_len)

        x_fractured = torch.chunk(x, self.num_devices, dim=-1)
        xs, rot_mats, attn_masks = [], [], []
        for i in range(self.num_devices):
            device = self.devices[i]
            # TODO: put input onto device sharded
            xs.append(
                torch2tt_tensor(
                    x_fractured[i],
                    device,
                    tt_layout=tt_lib.tensor.Layout.TILE,
                    tt_memory_config=self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"],
                    tt_dtype=self.model_config["WORD_EMBEDDING_OUTPUT_DTYPE"],
                )
            )
            rot_mats.append(
                torch2tt_tensor(rot_mat.clone(), device, tt_memory_config=self.model_config["ROT_MAT_MEMCFG"])
            )

            # Put attn_mask on the device with the sharded config
            attention_mask_memconfig = self.model_config["ATTN_MASK_MEMCFG"]
            if attention_mask_memconfig.is_sharded():
                attn_mask_shard_shape = attention_mask_memconfig.shard_spec.shape
                attn_mask_shard_shape[-1] = padded_layer_past_len
                attention_mask_memconfig.shard_spec.shape = attn_mask_shard_shape
            attn_masks.append(
                torch2tt_tensor(
                    attn_mask.clone(),
                    device,
                    tt_memory_config=attention_mask_memconfig,
                    tt_dtype=self.model_config["ATTN_MASK_DTYPE"],
                )
            )
        return (
            xs,
            start_pos,
            rot_mats,
            attn_masks,
        )

    def __call__(
        self,
        xs: list,
        rot_mats: list,
        start_pos: int,
        attn_masks: list,
    ) -> tt_lib.tensor.Tensor:
        ### xs (residual stream) is fractured on all chips
        xs_replicated = []
        # Put xs back on DRAM and do allgather
        for i in range(self.num_devices):
            xs_replicated.append(
                tt_lib.tensor.sharded_to_interleaved(xs[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"])
            )
        ### Duplicate inputs for layernorm
        if self.emulated:
            xs_replicated = tt_all_gather_torch(xs_replicated, dim=-1)
        else:
            xs_replicated = tt_lib.tensor.all_gather(
                xs_replicated,
                dim=3,
                num_links=1,
                output_mem_config=self.model_config["DEFAULT_MEMCFG"],
            )

        for i in range(self.num_devices):
            # RMSNorm must execute on sharded input
            xs_replicated[i] = tt_lib.tensor.interleaved_to_sharded(
                xs_replicated[i], sharded_mem_config=self.model_config["DECODER_ALL_GATHER_OUTPUT_MEMCFG"]
            )
        attn_norm_replicated = []
        for i in range(self.num_devices):
            attn_norm_replicated.append(
                tt_lib.operations.primary.rmsnorm(
                    xs_replicated[i],
                    self.norm_eps,
                    self.attn_norm_list[i],
                    program_config=self.model_config["LN_ATTN_PROGCFG"],
                    output_mem_config=self.model_config["LN_ATTN_OUTPUT_MEMCFG"],
                )
            )  # attn_norm_replicated is sharded
            # xs_replicated[i].deallocate(True) # Fine to deallocate here, because this is replicated from the sharded
        # attn_outs is fractured
        attn_outs = self.attention(attn_norm_replicated, rot_mats, start_pos, attn_masks)

        ### Fractured residual add
        # Add attn output to residiual first in place to save memory
        # Note that this is only correct in inference when dropout is disabled
        output = []
        residual = xs
        for i in range(self.num_devices):
            output.append(
                tt_lib.tensor.add_without_autoformat(
                    residual[i],
                    attn_outs[i],
                    output_mem_config=self.model_config["PARALLEL_ATTN_ADD_OUTPUT_MEMCFG"],
                    in_place=True,
                )
            )
            attn_outs[i].deallocate(True)

        attn_resid_replicated = []
        for i in range(self.num_devices):
            # Put attn_resid back on DRAM
            attn_resid_replicated.append(
                tt_lib.tensor.sharded_to_interleaved(output[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"])
            )

        ### Duplicate attention residual on all chips
        if self.emulated:
            attn_resid_replicated = tt_all_gather_torch(attn_resid_replicated, dim=-1)
        else:
            attn_resid_replicated = tt_lib.tensor.all_gather(
                attn_resid_replicated,
                dim=3,
                num_links=1,
                output_mem_config=self.model_config["DEFAULT_MEMCFG"],
            )

        for i in range(self.num_devices):
            # RMSNorm must execute on sharded input
            attn_resid_replicated[i] = tt_lib.tensor.interleaved_to_sharded(
                attn_resid_replicated[i], sharded_mem_config=self.model_config["DECODER_ALL_GATHER_OUTPUT_MEMCFG"]
            )
        ### Duplicate FFN layernorm
        ffn_norm_replicated = []
        for i in range(self.num_devices):
            ffn_norm_replicated.append(
                tt_lib.operations.primary.rmsnorm(
                    attn_resid_replicated[i],
                    self.norm_eps,
                    self.ffn_norm_list[i],
                    program_config=self.model_config["LN_MLP_PROGCFG"],
                    output_mem_config=self.model_config["PADDED_LN_MLP_OUTPUT_MEMCFG"],
                )
            )  # ffn_norm_replicated is sharded
        # attn_resid_replicated[i].deallocate(True) # !!! Cant deallocate here, will drop PCC

        ffn_out = self.mlp(ffn_norm_replicated)

        ### Dropout_add residual in place
        for i in range(self.num_devices):
            output[i] = tt_lib.tensor.add_without_autoformat(
                output[i],
                ffn_out[i],
                output_mem_config=self.model_config["DROPOUT_ADD_OUTPUT_MEMCFG"],
                in_place=True,
            )
            ffn_out[i].deallocate(True)

        # FOR BRINGUP! Outputs are sharded. Interleave them
        # for i in range(self.num_devices):
        #     output[i] = tt_lib.tensor.sharded_to_interleaved(
        #         output[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"]
        #     )
        return output
