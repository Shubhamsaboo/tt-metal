// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "debug/assert.h"
#include "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_ring_gather_utils.hpp"

void kernel_main() {
    // TODO: Update the interleaver receive reader kernel invocation to just be able to use this
    constexpr uint32_t num_chip_payloads = get_compile_time_arg_val(0);

    // Put this here for simpler merge with non-sharded kernel
    constexpr uint32_t num_transfers = num_chip_payloads;

    // Number of cores in the local chip serviced by this worker
    constexpr uint32_t num_cores_per_chip_payload = get_compile_time_arg_val(1);

    // For non-sharded kernels, hardcode to 1 so we can reuse the kernel
    constexpr uint32_t num_payloads_per_chip = num_cores_per_chip_payload;

    constexpr uint32_t num_pages_per_core_payload = get_compile_time_arg_val(2);
    constexpr uint32_t num_pages_per_eth_buffer = get_compile_time_arg_val(3);

    // For later merge with non-sharded kernels
    constexpr uint32_t num_full_chunks = num_pages_per_core_payload / num_pages_per_eth_buffer;
    constexpr uint32_t rem_num_pages = num_pages_per_core_payload % num_pages_per_eth_buffer;

    constexpr uint32_t page_size = get_compile_time_arg_val(4);

    // Info about the eth receiver eth core (producer of this core)
    constexpr uint32_t eth_receiver_noc_x = get_compile_time_arg_val(5);
    constexpr uint32_t eth_receiver_noc_y = get_compile_time_arg_val(6);
    constexpr uint32_t eth_receiver_l1_semaphore_addr = get_compile_time_arg_val(7);
    // TODO: Make this arch agnostic
    ASSERT(eth_receiver_noc_x > 1 && eth_receiver_noc_x < 12  && (eth_receiver_noc_y == 0 || eth_receiver_noc_y == 6);

    const uint32_t eth_receiver_l1_base_addr = get_arg_val<uint32_t>(0);
    const uint32_t receiver_read_sem_addr = get_arg_val<uint32_t>(1);

    // Eth receiver will set this semaphore when data is available
    volatile tt_l1_ptr uint32_t* receiver_read_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_read_sem_addr);

    // Address of the buffer on the eth receiver, this is different per receiver worker core
    const uint64_t eth_receiver_l1_base_noc_addr = get_noc_addr(eth_receiver_noc_x, eth_receiver_noc_y, eth_receiver_l1_base_addr);

    // Address of the semaphore on the eth receiver, this is the same per receiver worker core
    const uint64_t eth_receiver_l1_semaphore_noc_addr = get_noc_addr(eth_receiver_noc_x, eth_receiver_noc_y, eth_receiver_l1_semaphore_addr);

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;
    uint32_t transfers_completed = 0;

    for (uint32_t i = 0; i < num_transfers; ++i) {
        for (uint32_t p = 0; p < num_payloads_per_chip; p++) {
            if constexpr (num_full_chunks > 0) {
                for (uint32_t c = 0; c < num_full_chunks; ++c) {
                    uint64_t eth_receiver_l1_curr_noc_addr = eth_receiver_l1_base_noc_addr;
                    // DPRINT << "rwr " << ID << " semaphore_wait at " << (uint32_t)receiver_read_semaphore_addr_ptr << "\n";
                    noc_semaphore_wait(receiver_read_semaphore_addr_ptr, 1);
                    noc_semaphore_set(receiver_read_semaphore_addr_ptr, 0);
                    // Read page by page so that writer can be kicked off instead of being blocked waiting for full chunk to be read
                    // Look into perf/optimizations for this
                    // DPRINT << "rwr fetch_chunk\n";
                    fetch_chunk_sharded(cb_id_in0, num_pages_per_eth_buffer, page_size, eth_receiver_l1_base_noc_addr);
                    noc_semaphore_inc(eth_receiver_l1_semaphore_noc_addr, 1);
                    transfers_completed++;
                    // DPRINT << "rwr " << ID << " transfers_completed: " << transfers_completed << "\n";
                }
            }
            if constexpr (rem_num_pages > 0) {
                uint64_t eth_receiver_l1_curr_noc_addr = eth_receiver_l1_base_noc_addr;
                // DPRINT << "rwr " << ID << " semaphore_wait\n";
                noc_semaphore_wait(receiver_read_semaphore_addr_ptr, 1);
                noc_semaphore_set(receiver_read_semaphore_addr_ptr, 0);
                // DPRINT << "rwr " << ID << " fetch_chunk\n";
                fetch_chunk_sharded(cb_id_in0, rem_num_pages, page_size, eth_receiver_l1_base_noc_addr);
                noc_semaphore_inc(eth_receiver_l1_semaphore_noc_addr, 1);
                transfers_completed++;
                // DPRINT << "rwr " << ID << " transfers_completed: " << transfers_completed << "\n";
            }
        }
    }

    DPRINT << "rwr sharded DONE\n";
}
