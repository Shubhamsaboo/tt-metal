// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/kernels/cq_dispatcher.hpp"
#include "debug/dprint.h"

void kernel_main() {
    bool db_buf_switch = false;

    constexpr uint32_t host_finish_addr = get_compile_time_arg_val(0);
    constexpr uint32_t cmd_base_address = get_compile_time_arg_val(1);
    constexpr uint32_t consumer_data_buffer_size = get_compile_time_arg_val(2);

    volatile uint32_t* db_semaphore_addr = reinterpret_cast<volatile uint32_t*>(SEMAPHORE_BASE);

    uint64_t producer_noc_encoding = uint64_t(NOC_XY_ENCODING(PRODUCER_NOC_X, PRODUCER_NOC_Y)) << 32;
    uint64_t consumer_noc_encoding = uint64_t(NOC_XY_ENCODING(my_x[0], my_y[0])) << 32;

    while (true) {
        // Wait for producer to supply a command
        uint32_t command_start_addr = get_command_slot_addr<cmd_base_address, consumer_data_buffer_size>(db_buf_switch);
        uint32_t program_transfer_start_addr = command_start_addr + ((DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER + DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION * DeviceCommand::NUM_POSSIBLE_BUFFER_TRANSFERS) * sizeof(uint32_t));
        volatile tt_l1_ptr uint32_t* command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(command_start_addr);
        volatile tt_l1_ptr CommandHeader* header = (CommandHeader*)command_ptr;
        DPRINT << "ACQUIRE" << ENDL();
        db_acquire(db_semaphore_addr, consumer_noc_encoding);
        uint32_t num_workers = header->num_workers;
        uint32_t num_pages = header->num_pages;
        uint32_t producer_consumer_transfer_num_pages = header->producer_consumer_transfer_num_pages;

        DPRINT << "NUM PAGES: " << num_pages << ENDL();


        db_cb_config_t* db_cb_config = get_local_db_cb_config(CQ_CONSUMER_CB_BASE, db_buf_switch);
        const db_cb_config_t* remote_db_cb_config = get_remote_db_cb_config(CQ_CONSUMER_CB_BASE, db_buf_switch);
        uint32_t completion_data_size = header->completion_data_size;
        reset_dispatch_message_addr();
        write_and_launch_program(
            db_cb_config,
            remote_db_cb_config,
            (CommandHeader*)command_ptr,
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(program_transfer_start_addr),
            producer_noc_encoding,
            producer_consumer_transfer_num_pages);
        wait_for_program_completion(num_workers);

        // notify producer that it has completed a command
        noc_semaphore_inc(producer_noc_encoding | get_semaphore(0), 1);
        noc_async_write_barrier(); // Barrier for now

        db_buf_switch = not db_buf_switch;
        DPRINT << "DONE PROGRAM" << ENDL();
    }
}
