// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "risc_attribs.h"
#include "tt_metal/hostdevcommon/common_values.hpp"
#include "tt_metal/impl/dispatch/kernels/command_queue_common.hpp"
#include "cq_cmds.hpp"

CQReadInterface cq_read_interface;

inline __attribute__((always_inline)) volatile uint32_t* get_cq_issue_read_ptr() {
    return reinterpret_cast<volatile uint32_t*>(CQ_ISSUE_READ_PTR);
}

inline __attribute__((always_inline)) volatile uint32_t* get_cq_issue_write_ptr() {
    return reinterpret_cast<volatile uint32_t*>(CQ_ISSUE_WRITE_PTR);
}

// Only the read interface is set up on the device... the write interface
// belongs to host
FORCE_INLINE
void setup_issue_queue_read_interface(const uint32_t issue_region_rd_ptr, const uint32_t issue_region_size) {
    cq_read_interface.issue_fifo_rd_ptr = issue_region_rd_ptr >> 4;
    cq_read_interface.issue_fifo_size = issue_region_size >> 4;
    cq_read_interface.issue_fifo_limit = (issue_region_rd_ptr + issue_region_size) >> 4;
    cq_read_interface.issue_fifo_rd_toggle = 0;
}

template <uint32_t num_command_slots>
FORCE_INLINE void wait_consumer_idle(volatile tt_l1_ptr uint32_t* db_semaphore_addr) {
    while (*db_semaphore_addr != num_command_slots);
}

FORCE_INLINE
void wait_consumer_space_available(volatile tt_l1_ptr uint32_t* db_semaphore_addr) {
    while (*db_semaphore_addr == 0);
}

FORCE_INLINE
void update_producer_consumer_sync_semaphores(
    uint64_t producer_noc_encoding,
    uint64_t consumer_noc_encoding,
    volatile tt_l1_ptr uint32_t* producer_db_semaphore_addr,
    uint32_t consumer_db_semaphore) {
    // Decrement the semaphore value
    noc_semaphore_inc(producer_noc_encoding | uint32_t(producer_db_semaphore_addr), -1);  // Two's complement addition

    // Notify the consumer
    noc_semaphore_inc(consumer_noc_encoding | consumer_db_semaphore, 1);
    noc_async_write_barrier();  // Barrier for now
}

FORCE_INLINE
void issue_queue_wait_front() {
    DEBUG_STATUS('N', 'Q', 'W');
    uint32_t issue_write_ptr_and_toggle;
    uint32_t issue_write_ptr;
    uint32_t issue_write_toggle;
    do {
        issue_write_ptr_and_toggle = *get_cq_issue_write_ptr();
        issue_write_ptr = issue_write_ptr_and_toggle & 0x7fffffff;
        issue_write_toggle = issue_write_ptr_and_toggle >> 31;
    } while (cq_read_interface.issue_fifo_rd_ptr == issue_write_ptr and cq_read_interface.issue_fifo_rd_toggle == issue_write_toggle);
    DEBUG_STATUS('N', 'Q', 'D');
}

template <uint32_t host_issue_queue_read_ptr_addr>
FORCE_INLINE
void notify_host_of_issue_queue_read_pointer() {
    // These are the PCIE core coordinates
    constexpr static uint64_t pcie_address = (uint64_t(NOC_XY_ENCODING(PCIE_NOC_X, PCIE_NOC_Y)) << 32) | host_issue_queue_read_ptr_addr;
    uint32_t issue_rd_ptr_and_toggle = cq_read_interface.issue_fifo_rd_ptr | (cq_read_interface.issue_fifo_rd_toggle << 31);
    volatile tt_l1_ptr uint32_t* issue_rd_ptr_addr = get_cq_issue_read_ptr();
    issue_rd_ptr_addr[0] = issue_rd_ptr_and_toggle;
    noc_async_write(CQ_ISSUE_READ_PTR, pcie_address, 4);
    noc_async_write_barrier();
}

template <uint32_t host_issue_queue_read_ptr_addr>
FORCE_INLINE
void issue_queue_pop_front(uint32_t cmd_size_B) {
    // First part of equation aligns to nearest multiple of 32, and then we shift to make it a 16B addr. Both
    // host and device are consistent in updating their pointers in this way, so they won't get out of sync. The
    // alignment is necessary because we can only read/write from/to 32B aligned addrs in host<->dev communication.
    uint32_t cmd_size_16B = align(cmd_size_B, 32) >> 4;
    cq_read_interface.issue_fifo_rd_ptr += cmd_size_16B;
    if (cq_read_interface.issue_fifo_rd_ptr >= cq_read_interface.issue_fifo_limit) {
        cq_read_interface.issue_fifo_rd_ptr -= cq_read_interface.issue_fifo_size;
        cq_read_interface.issue_fifo_rd_toggle = not cq_read_interface.issue_fifo_rd_toggle;
    }

    notify_host_of_issue_queue_read_pointer<host_issue_queue_read_ptr_addr>();
}

FORCE_INLINE
void program_local_cb(uint32_t data_section_addr, uint32_t num_pages, uint32_t page_size, uint32_t cb_size) {
    uint32_t cb_id = 0;
    uint32_t fifo_addr = data_section_addr >> 4;
    uint32_t fifo_limit = fifo_addr + (cb_size >> 4);
    cb_interface[cb_id].fifo_limit = fifo_limit;  // to check if we need to wrap
    cb_interface[cb_id].fifo_wr_ptr = fifo_addr;
    cb_interface[cb_id].fifo_rd_ptr = fifo_addr;
    cb_interface[cb_id].fifo_size = cb_size >> 4;
    cb_interface[cb_id].tiles_acked = 0;
    cb_interface[cb_id].tiles_received = 0;
    cb_interface[cb_id].fifo_num_pages = num_pages;
    cb_interface[cb_id].fifo_page_size = page_size >> 4;
}

template <uint32_t producer_cmd_base_addr, uint32_t producer_data_buffer_size, uint32_t consumer_cmd_base_addr, uint32_t consumer_data_buffer_size>
FORCE_INLINE
void program_consumer_cb(
    db_cb_config_t* db_cb_config,
    const db_cb_config_t* remote_db_cb_config,
    bool db_buf_switch,
    uint64_t consumer_noc_encoding,
    uint32_t num_pages,
    uint32_t page_size,
    uint32_t cb_size) {
    /*
        This API programs the double-buffered CB space of the consumer. This API should be called
        before notifying the consumer that data is available.
    */
    uint32_t cb_start_rd_addr = get_db_buf_addr<producer_cmd_base_addr, producer_data_buffer_size>(db_buf_switch);
    uint32_t cb_start_wr_addr = get_db_buf_addr<consumer_cmd_base_addr, consumer_data_buffer_size>(db_buf_switch);

    db_cb_config->ack = 0;
    db_cb_config->recv = 0;
    db_cb_config->num_pages = num_pages;
    db_cb_config->page_size_16B = page_size >> 4;
    db_cb_config->total_size_16B = cb_size >> 4;
    db_cb_config->rd_ptr_16B = cb_start_rd_addr >> 4;
    db_cb_config->wr_ptr_16B = cb_start_wr_addr >> 4;

    noc_async_write(
        (uint32_t)(db_cb_config), consumer_noc_encoding | (uint32_t)(remote_db_cb_config), sizeof(db_cb_config_t));
    noc_async_write_barrier();  // barrier for now
}

FORCE_INLINE
bool cb_producer_space_available(int32_t num_pages) {
    uint32_t operand = 0;
    uint32_t pages_acked_ptr = (uint32_t) get_cb_tiles_acked_ptr(operand);

    // while the producer (write-side interface) is waiting for space to free up "tiles_pushed" is not changing
    // "tiles_pushed" is updated by the producer only when the tiles are pushed
    uint32_t pages_received = get_cb_tiles_received_ptr(operand)[0];

    int32_t free_space_pages;
    DEBUG_STATUS('C', 'R', 'B', 'W');

    // uint16_t's here because Tensix updates the val at tiles_acked_ptr as uint16 in llk_pop_tiles
    // TODO: I think we could have TRISC update tiles_acked_ptr, and we wouldn't need uint16 here
    uint16_t pages_acked = (uint16_t)reg_read(pages_acked_ptr);
    uint16_t free_space_pages_wrap =
        cb_interface[operand].fifo_num_pages - (pages_received - pages_acked);
    free_space_pages = (int32_t)free_space_pages_wrap;
    return free_space_pages >= num_pages;
}

FORCE_INLINE
bool cb_consumer_space_available(db_cb_config_t* db_cb_config, int32_t num_pages) {
    // TODO: delete cb_consumer_space_available and use this one

    DEBUG_STATUS('C', 'R', 'B', 'W');

    uint16_t free_space_pages_wrap = db_cb_config->num_pages - (db_cb_config->recv - db_cb_config->ack);
    int32_t free_space_pages = (int32_t)free_space_pages_wrap;
    DEBUG_STATUS('C', 'R', 'B', 'D');

    return free_space_pages >= num_pages;
}

template <uint32_t cmd_base_addr, uint32_t consumer_cmd_base_addr, uint32_t consumer_data_buffer_size>
FORCE_INLINE void relay_command(bool db_buf_switch, uint64_t consumer_noc_encoding) {
    /*
        Relays the current command to the consumer.
    */

    uint64_t consumer_command_slot_addr = consumer_noc_encoding | get_command_slot_addr<consumer_cmd_base_addr, consumer_data_buffer_size>(db_buf_switch);
    noc_async_write(cmd_base_addr, consumer_command_slot_addr, DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND);
    noc_async_write_barrier();
}

template <uint32_t consumer_cmd_base_addr, uint32_t consumer_data_buffer_size>
void produce(
    volatile tt_l1_ptr uint32_t* command_ptr,
    uint32_t num_srcs,
    bool sharded,
    uint32_t sharded_buffer_num_cores,
    uint32_t producer_cb_size,
    uint32_t producer_cb_num_pages,
    uint64_t consumer_noc_encoding,
    uint32_t producer_consumer_transfer_num_pages,
    bool db_buf_switch,
    db_cb_config_t* db_cb_config,
    const db_cb_config_t* remote_db_cb_config) {
    /*
        This API prefetches data from host memory and writes data to the consumer core. On the consumer,
        we partition the data space into 2 via double-buffering. There are two command slots, and two
        data slots. The producer reads in data into its local buffer and checks whether it can write to
        the consumer. It continues like this in a loop, context switching between pulling in data and
        writing to the consumer.
    */
    uint32_t consumer_cb_size = (db_cb_config->total_size_16B << 4);
    uint32_t consumer_cb_num_pages = db_cb_config->num_pages;

    command_ptr += DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER;
    uint32_t l1_consumer_fifo_limit_16B =
        (get_db_buf_addr<consumer_cmd_base_addr, consumer_data_buffer_size>(db_buf_switch) + consumer_cb_size) >> 4;


    for (uint32_t i = 0; i < num_srcs; i++) {
        const uint32_t bank_base_address = command_ptr[0];
        const uint32_t num_pages = command_ptr[2];
        const uint32_t page_size = command_ptr[3];
        const uint32_t src_buf_type = command_ptr[4];
        const uint32_t src_page_index = command_ptr[6];


        uint32_t fraction_of_producer_cb_num_pages = consumer_cb_num_pages / 2;

        uint32_t num_to_read = min(num_pages, fraction_of_producer_cb_num_pages);
        uint32_t num_to_write = min(num_pages, producer_consumer_transfer_num_pages); // This must be a bigger number for perf.
        uint32_t num_reads_issued = 0;
        uint32_t num_reads_completed = 0;
        uint32_t num_writes_completed = 0;
        uint32_t src_page_id = src_page_index;

        Buffer buffer;
        if ((BufferType)src_buf_type == BufferType::SYSTEM_MEMORY or not(sharded)) {
            buffer.init((BufferType)src_buf_type, bank_base_address, page_size);
        }
        else{
            buffer.init_sharded(page_size, sharded_buffer_num_cores, bank_base_address,
                            command_ptr + COMMAND_PTR_SHARD_IDX);
        }

        while (num_writes_completed != num_pages) {
            // Context switch between reading in pages and sending them to the consumer.
            // These APIs are non-blocking to allow for context switching.
            if (cb_producer_space_available(num_to_read) and num_reads_issued < num_pages) {
                uint32_t l1_write_ptr = get_write_ptr(0);
                buffer.noc_async_read_buffer(l1_write_ptr, src_page_id, num_to_read);
                cb_push_back(0, num_to_read);
                num_reads_issued += num_to_read;
                src_page_id += num_to_read;

                uint32_t num_pages_left = num_pages - num_reads_issued;
                num_to_read = min(num_pages_left, fraction_of_producer_cb_num_pages);
            }

            if (num_reads_issued > num_writes_completed and cb_consumer_space_available(db_cb_config, num_to_write)) {
                if (num_writes_completed == num_reads_completed) {
                    noc_async_read_barrier();
                    num_reads_completed = num_reads_issued;
                }

                uint32_t dst_addr = (db_cb_config->wr_ptr_16B << 4);
                uint64_t dst_noc_addr = consumer_noc_encoding | dst_addr;
                uint32_t l1_read_ptr = get_read_ptr(0);
                noc_async_write(l1_read_ptr, dst_noc_addr, page_size * num_to_write);
                multicore_cb_push_back(
                    db_cb_config, remote_db_cb_config, consumer_noc_encoding, l1_consumer_fifo_limit_16B, num_to_write);
                noc_async_write_barrier();
                cb_pop_front(0, num_to_write);
                num_writes_completed += num_to_write;
                num_to_write = min(num_pages - num_writes_completed, producer_consumer_transfer_num_pages);
            }
        }
        command_ptr += DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION;
    }
}

template <bool forward_path, uint32_t consumer_cmd_base_addr, uint32_t consumer_data_buffer_size>
void produce_for_eth_src_router(
    volatile tt_l1_ptr uint32_t* command_ptr,
    uint32_t num_srcs,
    uint32_t sharded_buffer_num_cores,
    uint32_t producer_cb_size,
    uint32_t producer_cb_num_pages,
    uint64_t eth_consumer_noc_encoding,
    uint32_t producer_consumer_transfer_num_pages,
    bool db_buf_switch,
    db_cb_config_t* db_cb_config,
    const db_cb_config_t* eth_db_cb_config) {
    /*
        This API prefetches data from host memory and writes data to the an ethernet core that relays the command
        to a remote chip, which will have a corresponding remote processor to consume the command.
    */
    uint32_t consumer_cb_size = (db_cb_config->total_size_16B << 4);
    uint32_t consumer_cb_num_pages = db_cb_config->num_pages;

    command_ptr += DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER;
    uint32_t l1_consumer_fifo_limit_16B =
        (get_db_buf_addr<consumer_cmd_base_addr, consumer_data_buffer_size>(db_buf_switch) + consumer_cb_size) >> 4;

    bool sharded = sharded_buffer_num_cores > 1;

    for (uint32_t i = 0; i < num_srcs; i++) {
        const uint32_t bank_base_address = command_ptr[0];
        const uint32_t num_pages = command_ptr[2];
        const uint32_t page_size = command_ptr[3];
        const uint32_t src_buf_type = command_ptr[4];
        const uint32_t src_page_index = command_ptr[6];

        bool read_from_sysmem = (BufferType)src_buf_type == BufferType::SYSTEM_MEMORY;
        if (forward_path & !read_from_sysmem) {
            // Data needs to come from target remote device
            continue;
        }

        // uint32_t fraction_of_producer_cb_num_pages = consumer_cb_num_pages / 2;

        uint32_t num_to_read = min(num_pages, producer_consumer_transfer_num_pages);
        uint32_t num_to_write =
            min(num_pages, producer_consumer_transfer_num_pages);  // This must be a bigger number for perf.
        uint32_t num_reads_issued = 0;
        uint32_t num_reads_completed = 0;
        uint32_t num_writes_completed = 0;
        uint32_t src_page_id = src_page_index;

        Buffer buffer;
        if (read_from_sysmem or not(sharded)) {
            buffer.init((BufferType)src_buf_type, bank_base_address, page_size);
        } else {
            buffer.init_sharded(
                page_size, sharded_buffer_num_cores, bank_base_address, command_ptr + COMMAND_PTR_SHARD_IDX);
        }

        while (num_writes_completed != num_pages) {
            // Context switch between reading in pages and sending them to the consumer.
            // These APIs are non-blocking to allow for context switching.
            if (cb_producer_space_available(num_to_read) and num_reads_issued < num_pages) {
                uint32_t l1_write_ptr = get_write_ptr(0);
                buffer.noc_async_read_buffer(l1_write_ptr, src_page_id, num_to_read);
                cb_push_back(0, num_to_read);
                num_reads_issued += num_to_read;
                src_page_id += num_to_read;

                uint32_t num_pages_left = num_pages - num_reads_issued;
                num_to_read = min(num_pages_left, producer_consumer_transfer_num_pages);
            }

              if (num_reads_issued > num_writes_completed and cb_consumer_space_available(db_cb_config, num_to_write)) {
                if (num_writes_completed == num_reads_completed) {
                    noc_async_read_barrier();
                    num_reads_completed = num_reads_issued;
                }

                uint32_t dst_addr = (db_cb_config->wr_ptr_16B << 4);
                uint64_t dst_noc_addr = eth_consumer_noc_encoding | dst_addr;
                uint32_t l1_read_ptr = get_read_ptr(0);
                noc_async_write(l1_read_ptr, dst_noc_addr, page_size * num_to_write);
                multicore_cb_push_back(
                    db_cb_config,
                    eth_db_cb_config,
                    eth_consumer_noc_encoding,
                    l1_consumer_fifo_limit_16B,
                    num_to_write);
                noc_async_write_barrier();
                cb_pop_front(0, num_to_write);
                num_writes_completed += num_to_write;
                num_to_write = min(num_pages - num_writes_completed, producer_consumer_transfer_num_pages);
            }
        }
        command_ptr += DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION;
    }
}

template <bool forward_path>
void transfer(
    db_cb_config_t* rx_db_cb_config,
    db_cb_config_t* tx_db_cb_config,
    const db_cb_config_t* remote_producer_db_cb_config,
    const db_cb_config_t* remote_consumer_db_cb_config,
    volatile tt_l1_ptr uint32_t* command_ptr,
    uint32_t num_buffer_transfers,
    uint32_t page_size,
    uint32_t producer_cb_size,
    uint32_t l1_producer_fifo_limit_16B,
    uint64_t producer_noc_encoding,
    uint32_t consumer_cb_size,
    uint32_t l1_consumer_fifo_limit_16B,
    uint64_t consumer_noc_encoding,
    uint32_t producer_consumer_transfer_num_pages,
    uint32_t l1_fifo_limit_16B)
{
    /*
        This API sends data from circular buffer in its L1 and writes data to the consumer core. On the consumer,
        we partition the data space into 2 via double-buffering. There are two command slots, and two
        data slots. Callees to transfer receive data into their local buffer and then check whether they can write to
        the consumer. It continues like this in a loop, context switching between waiting to receive data and
        writing to the consumer.
    */

    volatile tt_l1_ptr CommandHeader* header = (CommandHeader*)command_ptr;
    bool is_program = header->is_program_buffer;

    command_ptr += DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER;

    for (uint32_t i = 0; i < num_buffer_transfers; i++) {
        const uint32_t bank_base_address = command_ptr[0];
        const uint32_t num_pages = command_ptr[2];
        const uint32_t page_size = command_ptr[3];
        const uint32_t src_buf_type = command_ptr[4];
        const uint32_t dst_buf_type = command_ptr[5];
        const uint32_t src_page_index = command_ptr[6];
        uint32_t src_page_id = src_page_index;

        bool skip_tx = forward_path & (BufferType)dst_buf_type == BufferType::SYSTEM_MEMORY & !is_program;
        if (skip_tx) {
            continue;
        }

        Buffer buffer;
        bool read_local_buffer = forward_path & (BufferType)src_buf_type != BufferType::SYSTEM_MEMORY & is_program;
        if (read_local_buffer) {
            buffer.init((BufferType)src_buf_type, bank_base_address, page_size);
        }

        uint32_t num_to_transfer = min(num_pages, producer_consumer_transfer_num_pages); // This must be a bigger number for perf.
        uint32_t num_transfers_completed = 0;

        while (num_transfers_completed != num_pages) {
            // Wait for data to be received in local CB
            uint32_t src_addr = (tx_db_cb_config->rd_ptr_16B) << 4;
            if (read_local_buffer) {
                buffer.noc_async_read_buffer(src_addr, src_page_id, num_to_transfer);
                noc_async_read_barrier();
            } else {
                multicore_cb_wait_front(rx_db_cb_config, num_to_transfer);
            }

            // Transfer data to consumer CB
            uint32_t dst_addr = (tx_db_cb_config->wr_ptr_16B << 4);
            uint64_t dst_noc_addr = consumer_noc_encoding | dst_addr;
            while (!cb_consumer_space_available(tx_db_cb_config, num_to_transfer));
            noc_async_write(src_addr, dst_noc_addr, page_size * num_to_transfer);
            multicore_cb_push_back( // this should be the other db config
                tx_db_cb_config,
                remote_consumer_db_cb_config,
                consumer_noc_encoding,
                l1_consumer_fifo_limit_16B,
                num_to_transfer);
            if (not read_local_buffer) {
                multicore_cb_pop_front(
                    rx_db_cb_config,
                    remote_producer_db_cb_config,
                    producer_noc_encoding,
                    l1_producer_fifo_limit_16B,
                    num_to_transfer,
                    rx_db_cb_config->page_size_16B);
            }
            noc_async_write_barrier();
            tx_db_cb_config->rd_ptr_16B += tx_db_cb_config->page_size_16B * num_to_transfer;
            if (tx_db_cb_config->rd_ptr_16B >= l1_fifo_limit_16B) {
                tx_db_cb_config->rd_ptr_16B -= rx_db_cb_config->total_size_16B;
            }
            num_transfers_completed += num_to_transfer;
            num_to_transfer = min(num_pages - num_transfers_completed, producer_consumer_transfer_num_pages);
        }
        command_ptr += DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION;
    }
}
