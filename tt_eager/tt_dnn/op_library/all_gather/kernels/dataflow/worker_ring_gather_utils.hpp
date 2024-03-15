// // SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
// //
// // SPDX-License-Identifier: Apache-2.0

// #include "dataflow_api.h"

// FORCE_INLINE void push_filler_pages_to_cb(const uint32_t& cb_id, uint32_t num_pages) {
//     cb_reserve_back(cb_id, num_pages);
//     cb_push_back(cb_id, num_pages);
// }
// FORCE_INLINE void pop_filler_pages_from_cb(const uint32_t& cb_id, uint32_t num_pages) {
//     cb_wait_front(cb_id, num_pages);
//     cb_pop_front(cb_id, num_pages);
// }


// FORCE_INLINE void fetch_chunk(const uint32_t& cb_id, const uint32_t& num_pages, const uint32_t& page_size, uint64_t remote_l1_read_addr, uint64_t eth_receiver_l1_semaphore_noc_addr) {
//     cb_reserve_back(cb_id, num_pages);
//     uint32_t l1_write_addr = get_write_ptr(cb_id);
//     noc_async_read(remote_l1_read_addr, l1_write_addr, page_size * num_pages);
//     noc_async_read_barrier();
//     noc_semaphore_inc(eth_receiver_l1_semaphore_noc_addr, 1);
//     cb_push_back(cb_id, num_pages);
// }

// FORCE_INLINE void send_chunk(const uint32_t& cb_id, const uint32_t& num_pages, const uint32_t& page_size, uint64_t remote_l1_write_addr, uint64_t eth_l1_sender_semaphore_addr) {
//     cb_wait_front(cb_id, num_pages);
//     uint32_t l1_read_addr = get_read_ptr(cb_id);
//     noc_async_write(l1_read_addr, remote_l1_write_addr, page_size * num_pages);
//     noc_semaphore_inc(eth_l1_sender_semaphore_addr, 1);
//     noc_async_write_barrier();
//     cb_pop_front(cb_id, num_pages);
// }

// template<typename AddrGen>
// FORCE_INLINE void write_and_send_chunk(uint32_t& output_page_idx, uint32_t& col_idx, uint32_t& row_idx, const uint32_t& cb_id, const AddrGen& d, const uint32_t num_cols, const uint32_t num_rows, const uint32_t& col_offset, const uint32_t& row_offset, const uint32_t& num_pages, const uint32_t& page_size, uint64_t remote_l1_write_addr, uint64_t eth_l1_sender_semaphore_addr) {
//     uint32_t l1_read_addr = get_read_ptr(cb_id);
//     cb_wait_front(cb_id, num_pages);
//     for (uint32_t i = 0; i < num_pages; ++i) {
//         noc_async_write(l1_read_addr, remote_l1_write_addr, page_size);
//         remote_l1_write_addr += page_size;
//         #ifdef RM_INTERLEAVED
//         uint64_t dst_noc_addr = get_noc_addr(output_page_idx, d);
//         noc_async_write(l1_read_addr, dst_noc_addr, page_size);
//         output_page_idx++;
//         row_idx++;
//         if (row_idx == num_rows) {
//             row_idx = 0;
//             output_page_idx += row_offset;
//         }
//         #elif defined TILE_INTERLEAVED
//         noc_async_write_tile(output_page_idx, d, l1_read_addr);
//         output_page_idx++;
//         col_idx++;
//         if (col_idx == num_cols) {
//             output_page_idx += col_offset;
//             col_idx = 0;
//             row_idx++;
//             if (row_idx == num_rows) {
//                 row_idx = 0;
//                 output_page_idx += row_offset;
//             }
//         }
//         #endif
//         l1_read_addr += page_size;
//     }
//     noc_semaphore_inc(eth_l1_sender_semaphore_addr, 1);
//     noc_async_write_barrier();
//     cb_pop_front(cb_id, num_pages);
// }


// template<typename AddrGen>
// FORCE_INLINE void write_chunk(uint32_t& output_page_idx, uint32_t& col_idx, uint32_t& row_idx, const uint32_t& cb_id, const AddrGen& d, const uint32_t& num_cols, const uint32_t& num_rows, const uint32_t& col_offset, const uint32_t& row_offset, const uint32_t& num_pages, const uint32_t& page_size, uint64_t worker_send_reader_semaphore_noc_addr) {
//     uint32_t l1_read_addr = get_read_ptr(cb_id);
//     cb_wait_front(cb_id, num_pages);
//     for (uint32_t i = 0; i < num_pages; ++i) {
//         #ifdef RM_INTERLEAVED
//         uint64_t dst_noc_addr = get_noc_addr(output_page_idx, d);
//         noc_async_write(l1_read_addr, dst_noc_addr, page_size);
//         output_page_idx++;
//         row_idx++;
//         if (row_idx == num_rows) {
//             row_idx = 0;
//             output_page_idx += row_offset;
//         }
//         #elif defined TILE_INTERLEAVED
//         noc_async_write_tile(output_page_idx, d, l1_read_addr);
//         output_page_idx++;
//         col_idx++;
//         if (col_idx == num_cols) {
//             output_page_idx += col_offset;
//             col_idx = 0;
//             row_idx++;
//             if (row_idx == num_rows) {
//                 row_idx = 0;
//                 output_page_idx += row_offset;
//             }
//         }
//         #endif
//         l1_read_addr += page_size;
//     }
//     noc_semaphore_inc(worker_send_reader_semaphore_noc_addr, 1);
//     noc_async_write_barrier();
//     cb_pop_front(cb_id, num_pages);
// }

// template<typename AddrGen>
// FORCE_INLINE void read_chunk(uint32_t& input_page_idx, const uint32_t& cb_id, const AddrGen& s, const uint32_t& num_pages, const uint32_t& page_size) {
//     const uint32_t end_read_idx = input_page_idx + num_pages;
//     cb_reserve_back(cb_id, num_pages);
//     uint32_t local_l1_read_addr = get_write_ptr(cb_id);
//     for (; input_page_idx < end_read_idx; ++input_page_idx) {
//         #ifdef RM_INTERLEAVED
//         uint64_t src_noc_addr = get_noc_addr(input_page_idx, s);
//         noc_async_read(src_noc_addr, local_l1_read_addr, page_size);
//         #elif defined TILE_INTERLEAVED
//         noc_async_read_tile(input_page_idx, s, local_l1_read_addr);
//         #endif
//         local_l1_read_addr += page_size;
//     }
//     noc_async_read_barrier();
//     cb_push_back(cb_id, num_pages);
// }

// template<typename AddrGen>
// FORCE_INLINE void read_chunk(uint32_t& input_page_idx, uint32_t& col_idx, uint32_t& row_idx, const uint32_t &cb_id, const AddrGen& s, const uint32_t& num_cols, const uint32_t& num_rows, const uint32_t& col_offset, const uint32_t& row_offset, const uint32_t& num_pages, const uint32_t& page_size) {
//     cb_reserve_back(cb_id, num_pages);
//     uint32_t local_l1_read_addr = get_write_ptr(cb_id);
//      for (uint32_t i = 0; i < num_pages; ++i) {
//         #ifdef RM_INTERLEAVED
//         uint64_t src_noc_addr = get_noc_addr(input_page_idx, s);
//         noc_async_read(src_noc_addr, local_l1_read_addr, page_size);
//         input_page_idx++;
//         row_idx++;
//         if (row_idx == num_rows) {
//             row_idx = 0;
//             input_page_idx += row_offset;
//         }
//         #elif defined TILE_INTERLEAVED
//         noc_async_read_tile(input_page_idx, s, local_l1_read_addr);
//         input_page_idx++;
//         col_idx++;
//         if (col_idx == num_cols) {
//             input_page_idx += col_offset;
//             col_idx = 0;
//             row_idx++;
//             if (row_idx == num_rows) {
//                 row_idx = 0;
//                 input_page_idx += row_offset;
//             }
//         }
//         #endif
//         local_l1_read_addr += page_size;
//     }
//     noc_async_read_barrier();
//     cb_push_back(cb_id, num_pages);
// }

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt_eager/tt_dnn/op_library/ccl/ccl_common.hpp"
#include "debug/assert.h"

enum ShardType : uint8_t {
    Width = 0,
    Height = 1,
    Block = 2
};

// TODO: See if we can enable a custom address generator
// for width sharding (but somehow avoid division)
// curr dest = index / n_chunks_per_dest_core
// curr_chunk = index % n_chunks_per_dest_core
// curr_address = get_noc_addr(
//   dest_workers[curr_chunk].x,
//   dest_workers[curr_chunk].y,
//   dest_worker_base_addresses[curr_chunk] + curr_core_chunk * intra_core_stride
// )
// ... maybe start with the above and just see where we land in terms of perf?
template <ShardType TYPE, typename ... Args>
struct ShardAddrGen final : ccl::ArgListConstructible<Args...> {

    static FORCE_INLINE void build_with_placement_new(ShardAddrGen& placement_new_address, const uint32_t arg_index) {
        uint32_t curr_arg_index = arg_index;

        uint32_t start_offset_into_dest_shard  = get_arg_val<uint32_t>(curr_arg_index++);
        uint32_t input_shard_size_in_bytes  = get_arg_val<uint32_t>(curr_arg_index++);
        uint32_t chunks_per_core_before_advance  = get_arg_val<uint32_t>(curr_arg_index++);
        uint32_t num_dest_cores  = get_arg_val<uint32_t>(curr_arg_index++);
        ccl::WorkerXY *dest_cores = get_arg_addr(curr_arg_index);
        curr_arg_index = num_dest_cores;
        uint32_t *dest_core_start_addresses  = get_arg_addr(curr_arg_index);
        curr_arg_index = num_dest_cores;

        new (&placement_new_address) ShardAddrGen(
            curr_arg_index - arg_index,
            start_offset_into_dest_shard,
            input_shard_size_in_bytes,
            chunks_per_core_before_advance,
            num_dest_cores,
            dest_cores,
            dest_core_start_addresses
        );

    }

    // This addr gen will dump all tiles from an input shard contiguously, and dump the
    // next input shard contiguously after it. This approach depends on a follow up
    //
    ShardAddrGen(
        uint16_t num_args_consumed,
        uint32_t start_offset_into_dest_shard,
        uint32_t input_shard_size_in_bytes,
        uint32_t chunks_per_core_before_advance,
        uint32_t num_dest_cores,
        ccl::WorkerXY *dest_cores,
        uint32_t *dest_core_start_addresses
        ) :
        ccl::ArgListConstructible<Args...>(num_args_consumed),
        start_offset_into_dest_shard(start_offset_into_dest_shard),
        input_shard_size_in_bytes(input_shard_size_in_bytes),
        chunks_per_core_before_advance(chunks_per_core_before_advance),
        num_dest_cores(num_dest_cores),
        dest_cores(dest_cores),
        dest_core_start_addresses(dest_core_start_addresses),
        curr_dest_worker_index(0),
        curr_worker_index(0),
        core_write_offset(0),
        curr_core_chunk_index(0),
        curr_core_write_offset(input_shard_size_in_bytes), // when cycling back to start core, this is the offset to resume writing at
        completed_core_wrap(false)
    {


    };
    static_assert(TYPE == ShardType::Width || TYPE == ShardType::Height || TYPE == ShardType::Block, "Invalid ShardType");

    // Clockwise vs counter clockwise only affects worker core traversal order (relative to canonical order). Since the dest
    // core list is a configurable list, we will, for now, require the host side kernel config code to produce the correc order
    // per worker
    void advance() {
        if constexpr (TYPE == ShardType::Width or TYPE == ShardType::Height) {
            // If I analyzed it properly, then we should never be wrapping back to the first dest core *and* still have
            // tiles/input shards to move
            ASSERT(!completed_core_wrap);
            this->curr_core_chunk_index++;
            this->curr_l1_address += this->intra_core_stride;
            bool do_chunk_wrap = this->curr_core_chunk_index == this->chunks_per_core_before_advance;
            if (do_chunk_wrap) {
                this->curr_core_chunk = 0;
                bool do_core_wrap = this->curr_worker_index == this->num_dest_cores - 1;
                if (do_core_wrap) {
                    this->curr_worker_index = 0;
                    completed_core_wrap = true;
                    this->core_write_offset += this->chunks_per_core_before_advance * this->intra_core_stride;
                } else {
                    this->curr_worker_index++;
                }
                this->curr_l1_address = this->dest_core_start_addresses[this->curr_worker_index] + this->core_write_offset;
            }
        } else {
            ASSERT(false);
            // Unsupported
        }
    }

    ccl::WorkerXY get_next_noc_xy_core() const {
        return this->dest_workers[this->curr_worker_index];
    }

    uint64_t get_next_noc_addr_and_advance() {
        if constexpr (TYPE == ShardType::Width) {
            ccl::WorkerXY dest_worker = dest_cores[curr_worker_index];
            uint32_t curr_address = this->curr_l1_address;
            this->advance();
            return get_noc_addr(dest_worker.x, dest_worker.y, curr_address);
        } else {
            ASSERT(false);
            // Unsupported
        }
    }


    // // Challenge is n_chunks_per_dest_core is not as reliably known (not convinced it can't
    // // be non-power-of-2)
    // // global index
    // // In this scheme dest_worker_base_addresses needs to contain all entries
    // // for all cores (256B) -> maybe that's fine
    // // offset = start_offset_into_dest_shard
    // uint64_t get_noc_addr(uint32_t index, uint32_t const offset=0) {
    //     uint32_t curr_dest_index = index / this->n_chunks_per_dest_core;
    //     // uint32_t curr_chunk = index % this->n_chunks_per_dest_core;
    //     uint32_t curr_core_chunk = index - (this->n_chunks_per_dest_core * curr_dest_index);
    //     uint32_t dest_worker_x = curr_dest_index & ((0x1 << 3) - 1);
    //     uint32_t dest_worker_y = curr_dest_index & ~((0x1 << 3) - 1);
    //     uint64_t curr_address = get_noc_addr(
    //       dest_worker_x,
    //       dest_worker_y,
    //       offset/*start_offset_into_dest_shard*/ +
    //         this->dest_worker_base_addresses[curr_dest_index] +
    //         curr_core_chunk * this->intra_core_stride;
    //     );
    //     return curr_address;
    // }

    uint32_t get_shard_size_in_bytes() const {
        return this->input_shard_size_in_bytes;
    }

    uint32_t get_num_dest_cores() const { return this->num_dest_cores; }
    uint32_t get_chunks_per_core_before_advance() const { return this->chunks_per_core_before_advance; }


    uint32_t start_offset_into_dest_shard;
    uint32_t input_shard_size_in_bytes;
    uint32_t chunks_per_core_before_advance;
    uint32_t num_dest_cores;
    ccl::WorkerXY *dest_cores;
    uint32_t *dest_core_start_addresses;
    uint32_t curr_dest_worker_index;
    uint32_t curr_worker_index;
    uint32_t core_write_offset;
    uint32_t curr_core_chunk_index;
    uint32_t curr_core_write_offset;
    uint8_t num_args_consumed;
    bool completed_core_wrap;

};

// template <ShardType TYPE>
// FORCE_INLINE std::uint64_t get_noc_addr(const uint32_t index, const ShardAddrGen<TYPE>& s, uint32_t offset = 0) {
//     // Assumes id is a global index into
//     return s.get_noc_addr(index, offset);
// }

FORCE_INLINE void fetch_chunk(const uint32_t& cb_id, const uint32_t& num_pages, const uint32_t& page_size, uint64_t remote_l1_read_addr) {
    for (uint32_t i = 0; i < num_pages; ++i) {
        cb_reserve_back(cb_id, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id);
        noc_async_read(remote_l1_read_addr, l1_write_addr, page_size);
        remote_l1_read_addr += page_size;
        noc_async_read_barrier();
        cb_push_back(cb_id, 1);
    }
}
// TODO(snijjar): Optimize this with the above when we enable the full-chunk `send_chunk` for non-sharded configs
FORCE_INLINE void fetch_chunk_sharded(const uint32_t& cb_id, const uint32_t& num_pages, const uint32_t& page_size, uint64_t remote_l1_read_addr) {
    cb_reserve_back(cb_id, num_pages);
    uint32_t l1_write_addr = get_write_ptr(cb_id);
    noc_async_read(remote_l1_read_addr, l1_write_addr, num_pages * page_size);
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}


FORCE_INLINE void send_chunk(const uint32_t& cb_id, const uint32_t& num_pages, const uint32_t& page_size, uint64_t remote_l1_write_addr) {
    for (uint32_t i = 0; i < num_pages; ++i) {
        cb_wait_front(cb_id, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id);
        noc_async_write(l1_read_addr, remote_l1_write_addr, page_size);
        remote_l1_write_addr += page_size;
        noc_async_write_barrier();
        cb_pop_front(cb_id, 1);
    }
}
// TODO(snijjar): Optimize this with the above when we enable the full-chunk `send_chunk` for non-sharded configs
FORCE_INLINE void send_chunk_sharded(const uint32_t& cb_id, const uint32_t& num_pages, const uint32_t& page_size, uint64_t remote_l1_write_addr) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    noc_async_write(l1_read_addr, remote_l1_write_addr, page_size * num_pages);
    // TODO(snijjar): move the semaphore inc here
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}



template <ShardType T>
FORCE_INLINE void write_and_send_chunk_sharded(const uint32_t& cb_id, ShardAddrGen<T> &addr_gen, uint32_t num_pages, uint64_t remote_eth_l1_write_addr) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    uint64_t dest_worker_noc_addr = addr_gen.get_next_noc_addr_and_advance();
    noc_async_write(l1_read_addr, remote_eth_l1_write_addr, addr_gen.get_shard_size_in_bytes());
    noc_async_write(l1_read_addr, dest_worker_noc_addr, addr_gen.get_shard_size_in_bytes());
    // TODO(snijjar): move the semaphore inc here
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}
template<typename AddrGen>
FORCE_INLINE void write_and_send_chunk(uint32_t& output_page_idx, uint32_t& col_idx, uint32_t& row_idx, const uint32_t& cb_id, const AddrGen& d, const uint32_t num_cols, const uint32_t num_rows, const uint32_t& col_offset, const uint32_t& row_offset, const uint32_t& num_pages, const uint32_t& page_size, uint64_t remote_l1_write_addr) {
    for (uint32_t i = 0; i < num_pages; ++i) {
        cb_wait_front(cb_id, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id);
        noc_async_write(l1_read_addr, remote_l1_write_addr, page_size);
        remote_l1_write_addr += page_size;
        #ifdef RM_INTERLEAVED
        uint64_t dst_noc_addr = get_noc_addr(output_page_idx, d);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size);
        output_page_idx++;
        row_idx++;
        if (row_idx == num_rows) {
            row_idx = 0;
            output_page_idx += row_offset;
        }
        #elif defined TILE_INTERLEAVED || defined SHARDED
        noc_async_write_tile(output_page_idx, d, l1_read_addr);
        output_page_idx++;
        col_idx++;
        if (col_idx == num_cols) {
            output_page_idx += col_offset;
            col_idx = 0;
            row_idx++;
            if (row_idx == num_rows) {
                row_idx = 0;
                output_page_idx += row_offset;
            }
        }
        #endif
        noc_async_write_barrier();
        cb_pop_front(cb_id, 1);
    }
}


template <ShardType T>
FORCE_INLINE void write_chunk_sharded(const uint32_t& cb_id, ShardAddrGen<T> &addr_gen, uint32_t num_pages) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    uint64_t dest_worker_noc_addr = addr_gen.get_next_noc_addr_and_advance();
    noc_async_write(l1_read_addr, dest_worker_noc_addr, addr_gen.get_shard_size_in_bytes());
    // TODO(snijjar): move the semaphore inc here
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}
template<typename AddrGen>
FORCE_INLINE void write_chunk(uint32_t& output_page_idx, uint32_t& col_idx, uint32_t& row_idx, const uint32_t& cb_id, const AddrGen& d, const uint32_t& num_cols, const uint32_t& num_rows, const uint32_t& col_offset, const uint32_t& row_offset, const uint32_t& num_pages, const uint32_t& page_size) {
    for (uint32_t i = 0; i < num_pages; ++i) {
        cb_wait_front(cb_id, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id);
        #ifdef RM_INTERLEAVED
        uint64_t dst_noc_addr = get_noc_addr(output_page_idx, d);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size);
        output_page_idx++;
        row_idx++;
        if (row_idx == num_rows) {
            row_idx = 0;
            output_page_idx += row_offset;
        }
        #elif defined TILE_INTERLEAVED
        noc_async_write_tile(output_page_idx, d, l1_read_addr);
        output_page_idx++;
        col_idx++;
        if (col_idx == num_cols) {
            output_page_idx += col_offset;
            col_idx = 0;
            row_idx++;
            if (row_idx == num_rows) {
                row_idx = 0;
                output_page_idx += row_offset;
            }
        }
        #endif
        noc_async_write_barrier();
        cb_pop_front(cb_id, 1);
    }
}


template <ShardType T>
FORCE_INLINE void read_chunk_from_input_tensor_sharded(const uint32_t& cb_id, ShardAddrGen<T> &addr_gen, uint32_t num_pages) {
    cb_reserve_back(cb_id, num_pages);
    uint32_t local_l1_read_dest_addr = get_write_ptr(cb_id);
    uint64_t src_noc_addr = addr_gen.get_next_noc_addr_and_advance();
    noc_async_read(src_noc_addr, local_l1_read_dest_addr, addr_gen.get_shard_size_in_bytes());
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}
// read chunk from input tensor (local chip)
template<typename AddrGen>
FORCE_INLINE void read_chunk_from_input_tensor(uint32_t& input_page_idx, const uint32_t& cb_id, const AddrGen& s, const uint32_t& num_pages, const uint32_t& page_size) {
    const uint32_t end_read_idx = input_page_idx + num_pages;
    for (; input_page_idx < end_read_idx; ++input_page_idx) {
        cb_reserve_back(cb_id, 1);
        uint32_t local_l1_read_addr = get_write_ptr(cb_id);
        #ifdef RM_INTERLEAVED
        uint64_t src_noc_addr = get_noc_addr(input_page_idx, s);
        noc_async_read(src_noc_addr, local_l1_read_addr, page_size);
        #elif defined TILE_INTERLEAVED
        noc_async_read_tile(input_page_idx, s, local_l1_read_addr);
        #endif
        noc_async_read_barrier();
        cb_push_back(cb_id, 1);
    }
}

// Same function - just different address generators? Commonize later
template <ShardType T>
FORCE_INLINE void read_chunk_from_output_tensor_sharded(const uint32_t& cb_id, ShardAddrGen<T> &addr_gen, uint32_t num_pages) {
    cb_reserve_back(cb_id, num_pages);
    uint32_t local_l1_read_dest_addr = get_write_ptr(cb_id);
    uint64_t src_noc_addr = addr_gen.get_next_noc_addr_and_advance();
    noc_async_read(src_noc_addr, local_l1_read_dest_addr, addr_gen.get_shard_size_in_bytes());
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}
// read chunk from output tensor (local chip)
template<typename AddrGen>
FORCE_INLINE void read_chunk_from_output_tensor(uint32_t& input_page_idx, uint32_t& col_idx, uint32_t& row_idx, const uint32_t &cb_id, const AddrGen& s, const uint32_t& num_cols, const uint32_t& num_rows, const uint32_t& col_offset, const uint32_t& row_offset, const uint32_t& num_pages, const uint32_t& page_size) {
     for (uint32_t i = 0; i < num_pages; ++i) {
        cb_reserve_back(cb_id, 1);
        uint32_t local_l1_read_addr = get_write_ptr(cb_id);
        #ifdef RM_INTERLEAVED
        uint64_t src_noc_addr = get_noc_addr(input_page_idx, s);
        noc_async_read(src_noc_addr, local_l1_read_addr, page_size);
        input_page_idx++;
        row_idx++;
        if (row_idx == num_rows) {
            row_idx = 0;
            input_page_idx += row_offset;
        }
        #elif defined TILE_INTERLEAVED
        noc_async_read_tile(input_page_idx, s, local_l1_read_addr);
        input_page_idx++;
        col_idx++;
        if (col_idx == num_cols) {
            input_page_idx += col_offset;
            col_idx = 0;
            row_idx++;
            if (row_idx == num_rows) {
                row_idx = 0;
                input_page_idx += row_offset;
            }
        }
        #endif
        noc_async_read_barrier();
        cb_push_back(cb_id, 1);
    }
}
