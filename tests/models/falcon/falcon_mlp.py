import torch
from torch import nn
import tt_lib

from models.helper_funcs import Linear as TTLinear
from models.utility_functions import torch2tt_tensor


class TtFalconMLP(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        model_config,
        tt_cache_path,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.hidden_size = hidden_size
        self.model_config = model_config

        layer_name = f"{base_url}.{layer_num}"

        dense_h_to_4h_str = f"{layer_name}.mlp.dense_h_to_4h.weight"
        dense_4h_to_h_str = f"{layer_name}.mlp.dense_4h_to_h.weight"

        if tt_cache_path is not None:
            self.dense_h_to_4h_weights = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{dense_h_to_4h_str}_{self.model_config['DENSE_H_TO_4H_MM_WEIGHTS_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_MEMCFG"])
            self.dense_4h_to_h_weights = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{dense_4h_to_h_str}_{self.model_config['DENSE_4H_TO_H_MM_WEIGHTS_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["DENSE_4H_TO_H_MM_WEIGHTS_MEMCFG"])
        else:
            r_dense_h_to_4h_weights = torch.rand(self.state_dict[dense_h_to_4h_str].shape)
            self.dense_h_to_4h_weights = torch2tt_tensor(
                torch.transpose(
                    r_dense_h_to_4h_weights,
                    -2,
                    -1,
                ),
                self.device,
                tt_memory_config=self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_MEMCFG"],
                tt_dtype=self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_DTYPE"],
            )
            # self.dense_h_to_4h_weights = torch2tt_tensor(
            #     torch.transpose(
            #         self.state_dict[dense_h_to_4h_str],
            #         -2,
            #         -1,
            #     ),
            #     self.device,
            #     tt_memory_config=self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_MEMCFG"],
            #     tt_dtype=self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_DTYPE"],
            # )
            r_dense_4h_to_h_weights = torch.rand(self.state_dict[dense_4h_to_h_str].shape)
            self.dense_4h_to_h_weights = torch2tt_tensor(
                torch.transpose(
                    r_dense_4h_to_h_weights,
                    -2,
                    -1,
                ),
                self.device,
                tt_memory_config=self.model_config["DENSE_4H_TO_H_MM_WEIGHTS_MEMCFG"],
                tt_dtype=self.model_config["DENSE_4H_TO_H_MM_WEIGHTS_DTYPE"],
            )
            # self.dense_4h_to_h_weights = torch2tt_tensor(
            #     torch.transpose(
            #         self.state_dict[dense_4h_to_h_str],
            #         -2,
            #         -1,
            #     ),
            #     self.device,
            #     tt_memory_config=self.model_config["DENSE_4H_TO_H_MM_WEIGHTS_MEMCFG"],
            #     tt_dtype=self.model_config["DENSE_4H_TO_H_MM_WEIGHTS_DTYPE"],
            # )

        # self.act_fn = tt_lib.tensor.gelu

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        hidden_states = tt_lib.tensor.falcon_dense_h_to_4h_matmul(
            x,
            self.dense_h_to_4h_weights,
            fuse_gelu_activation=True,
            output_mem_config=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_DTYPE"],
        )
        x.deallocate()

        hidden_states = tt_lib.tensor.falcon_dense_4h_to_h_matmul(
            hidden_states,
            self.dense_4h_to_h_weights,
            output_mem_config=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_DTYPE"],
        )

        # return TT Tensor
        return hidden_states
