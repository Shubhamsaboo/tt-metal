// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/kernels/command_queue_producer.hpp"
#include "debug/dprint.h"

// Receives fast dispatch packets from ethernet router and forwards them to dispatcher kernel
void kernel_main() {
    constexpr uint32_t cmd_base_addr = get_compile_time_arg_val(0);
    constexpr uint32_t data_section_addr = get_compile_time_arg_val(1);
    constexpr uint32_t data_buffer_size = get_compile_time_arg_val(2);
    constexpr uint32_t producer_cmd_base_addr = get_compile_time_arg_val(3);
    constexpr uint32_t producer_data_buffer_size = get_compile_time_arg_val(4);
    constexpr uint32_t dispatcher_cmd_base_addr = get_compile_time_arg_val(5);
    constexpr uint32_t dispatcher_data_buffer_size = get_compile_time_arg_val(6);

    // Initialize the producer/consumer DB semaphore
    // This represents how many buffers the producer can write to.
    uint64_t producer_noc_encoding = uint64_t(NOC_XY_ENCODING(PRODUCER_NOC_X, PRODUCER_NOC_Y)) << 32;
    uint64_t processor_noc_encoding = uint64_t(NOC_XY_ENCODING(my_x[0], my_y[0])) << 32;
    uint64_t dispatcher_noc_encoding = uint64_t(NOC_XY_ENCODING(DISPATCHER_NOC_X, DISPATCHER_NOC_Y)) << 32;

    volatile tt_l1_ptr uint32_t* rx_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));  // Should be initialized to 1 by host
    volatile tt_l1_ptr uint32_t* db_tx_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));  // Should be num command slots by in the dispatcher

    constexpr bool rx_buf_switch = false;   // atm only one slot to receive commands from ethernet
    bool db_tx_buf_switch = false;
    while (true) {
        // Wait for ethernet router to supply a command
        DPRINT << "remote cmd processor waiting for data" << ENDL();
        db_acquire(rx_semaphore_addr, processor_noc_encoding);

        // For each instruction, we need to jump to the relevant part of the device command
        uint32_t command_start_addr = get_command_slot_addr<cmd_base_addr, data_buffer_size>(rx_buf_switch);
        volatile tt_l1_ptr uint32_t* command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(command_start_addr);
        DPRINT << "command start addr " << command_start_addr << ENDL();

        uint32_t num_buffer_transfers = command_ptr[DeviceCommand::num_buffer_transfers_idx];
        uint32_t stall = command_ptr[DeviceCommand::stall_idx];
        uint32_t page_size = command_ptr[DeviceCommand::page_size_idx];
        uint32_t producer_cb_size = command_ptr[DeviceCommand::producer_cb_size_idx];
        uint32_t consumer_cb_size = command_ptr[DeviceCommand::consumer_cb_size_idx];
        uint32_t producer_cb_num_pages = command_ptr[DeviceCommand::producer_cb_num_pages_idx];
        uint32_t consumer_cb_num_pages = command_ptr[DeviceCommand::consumer_cb_num_pages_idx];
        uint32_t num_pages = command_ptr[DeviceCommand::num_pages_idx];
        uint32_t producer_consumer_transfer_num_pages = command_ptr[DeviceCommand::producer_consumer_transfer_num_pages_idx];

        // eth core should be programming the remote command processor CB?
        DPRINT << "remote cmd processor kernel got:"
               << " " << num_buffer_transfers
               << " " << page_size
               << " " << producer_cb_size
               << " " << consumer_cb_size
               << " " << producer_cb_num_pages
               << " " << consumer_cb_num_pages
               << " " << num_pages
               << " " << producer_consumer_transfer_num_pages
               << " " << stall << ENDL();

        DPRINT << "rcp: db_tx_semaphore_addr is " << get_semaphore(1) << " value is " << db_tx_semaphore_addr[0] << ENDL();
        while (db_tx_semaphore_addr[0] == 0)
            ;  // Check that there is space in the dispatcher
        DPRINT << "rcp: done waiting on db_tx_semaphore_addr" << ENDL();
        program_consumer_cb<dispatcher_cmd_base_addr, dispatcher_data_buffer_size>(db_tx_buf_switch, dispatcher_noc_encoding, consumer_cb_num_pages, page_size, consumer_cb_size);
        DPRINT << "rcp: dispatcher_cmd_base_addr: " << dispatcher_cmd_base_addr << ENDL();
        relay_command<dispatcher_cmd_base_addr, dispatcher_data_buffer_size>(db_tx_buf_switch, dispatcher_noc_encoding);
        if (stall) {
            while (*db_tx_semaphore_addr != 2)
                ;
        }

        // Decrement the semaphore value
        noc_semaphore_inc(producer_noc_encoding | uint32_t(db_tx_semaphore_addr), -1);  // Two's complement addition
        noc_async_write_barrier();

        // Notify the consumer
        noc_semaphore_inc(dispatcher_noc_encoding | get_semaphore(0), 1);
        noc_async_write_barrier();  // Barrier for now

        transfer(
            command_ptr,
            num_buffer_transfers,
            page_size,
            producer_cb_size,
            get_db_buf_addr<producer_cmd_base_addr, producer_data_buffer_size>(rx_buf_switch) + producer_cb_size,
            producer_noc_encoding,
            consumer_cb_size,
            get_db_buf_addr<dispatcher_cmd_base_addr, dispatcher_data_buffer_size>(db_tx_buf_switch) + consumer_cb_size,
            dispatcher_noc_encoding,
            producer_consumer_transfer_num_pages,
            rx_buf_switch,
            db_tx_buf_switch
        );

        // notify producer ethernet router that it has completed transferring a command
        noc_semaphore_inc(producer_noc_encoding | get_semaphore(0), 1);
        noc_async_write_barrier(); // Barrier for now

        db_tx_buf_switch = not db_tx_buf_switch;
    }
}
