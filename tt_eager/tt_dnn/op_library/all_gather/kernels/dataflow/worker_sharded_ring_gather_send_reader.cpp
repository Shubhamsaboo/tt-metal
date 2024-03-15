// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_ring_gather_utils.hpp"

void kernel_main() {
    constexpr ShardType shard_type = static_cast<ShardType>(get_compile_time_arg_val(0));
    constexpr uint32_t num_transfers = get_compile_time_arg_val(1);

    ShardAddrGen<shard_type> input_tensor_shard_reader;
    ShardAddrGen<shard_type> output_tensor_shard_reader;

    uint32_t arg_index = 0;
    volatile tt_l1_ptr uint32_t* local_semaphore_address = get_arg_val<volatile tt_l1_ptr uint32_t*>(arg_index++);
    input_tensor_shard_reader.build_with_placement_new(arg_index);
    arg_index += input_tensor_shard_reader.get_num_args_consumed();
    output_tensor_shard_reader.build_with_placement_new(arg_index);
    arg_index += output_tensor_shard_reader.get_num_args_consumed();

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;

    ASSERT(
        output_tensor_shard_reader.get_num_dest_core() * output_tensor_shard_reader.get_chunks_per_core_before_advance() ==
        num_transfers * num_chunks_per_transfer
    );

    uint32_t num_chunks_per_transfer = input_tensor_shard_reader.get_num_dest_core() * input_tensor_shard_reader.get_chunks_per_core_before_advance();

    for (uint32_t c = 0; c < num_chunks_per_transfer; ++c) {
        read_chunk_local_chip_input_sharded(cb_id_in0, input_tensor_shard_reader, 1);
    }

    uint32_t sem_idx = 1;

    for (uint32_t i = 1; i < num_transfers; ++i) {
        for (uint32_t c = 0; c < num_chunks_per_transfer; ++c) {
            noc_semaphore_wait_min(local_semaphore_address, sem_idx);
            sem_idx++;
            read_chunk_from_output_tensor(cb_id_in0, output_tensor_shard_reader, 1); // 1 chunk == 1 page?
        }
    }

}
