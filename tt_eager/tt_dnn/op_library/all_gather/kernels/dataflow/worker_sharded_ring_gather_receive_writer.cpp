// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_ring_gather_utils.hpp"


void kernel_main() {

    constexpr ShardType shard_type = static_cast<ShardType>(get_compile_time_arg_val(0));

    ShardAddrGen<shard_type> output_tensor_shard_writer;
    uint32_t arg_index = 0;
    uint32_t remote_sender_reader_semaphore_addres = get_arg_val<uint32_t>(arg_index++);
    uint32_t shards_per_ = get_arg_val<uint32_t>(arg_index++);
    output_tensor_shard_writer.build_with_placement_new(arg_index);
    arg_index += output_tensor_shard_writer.get_num_args_consumed();

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;
    // Each worker receiver writer matches with a specific worker sender reader
    // Used to signal that data has been committed to memory and can be read

    uint32_t num_transfers = output_tensor_shard_writer.get_num_dest_core() * output_tensor_shard_writer.get_chunks_per_core_before_advance();
    for (uint32_t d = 0; d < num_transfers; d++) {

        ccl::WorkerXY dest_worker_xy = output_tensor_shard_writer.get_next_noc_xy_core();
        write_chunk_sharded(cb_id_in0, output_tensor_shard_writer, 1); // 1 shard = 1 page?
        const uint64_t worker_send_reader_semaphore_noc_addr = get_noc_addr(dest_worker_xy.x, dest_worker_xy.y, remote_sender_reader_semaphore_addres);
        noc_semaphore_inc(worker_send_reader_semaphore_noc_addr, 1);
    }
}
