#include <stdint.h>
#include "dataflow_api.h"
#include "debug_print.h"

void kernel_main() {


    bool one_time_profile = true;

    // in0 tensor args
    uint32_t in0_tensor_addr                    = get_arg_val<uint32_t>(0);
    uint32_t in0_tensor_start_tile_id           = get_arg_val<uint32_t>(1);
    uint32_t in0_tensor_stride_w                = get_arg_val<uint32_t>(2);
    uint32_t in0_tensor_stride_h                = get_arg_val<uint32_t>(3);
    uint32_t in0_tensor_next_block_stride       = get_arg_val<uint32_t>(4);

    // in0 block args
    uint32_t in0_block_w                        = get_arg_val<uint32_t>(5);
    uint32_t in0_block_h                        = get_arg_val<uint32_t>(6);
    uint32_t in0_block_num_tiles                = get_arg_val<uint32_t>(7);

    // in1 tensor args
    uint32_t in1_tensor_addr                    = get_arg_val<uint32_t>(8);
    uint32_t in1_tensor_start_tile_id           = get_arg_val<uint32_t>(9);
    uint32_t in1_tensor_stride_w                = get_arg_val<uint32_t>(10);
    uint32_t in1_tensor_stride_h                = get_arg_val<uint32_t>(11);
    uint32_t in1_tensor_next_block_stride       = get_arg_val<uint32_t>(12);

    // in1 block args
    uint32_t in1_block_w                        = get_arg_val<uint32_t>(13);
    uint32_t in1_block_h                        = get_arg_val<uint32_t>(14);
    uint32_t in1_block_num_tiles                = get_arg_val<uint32_t>(15);

    // in0/in1 common args
    uint32_t num_blocks                         = get_arg_val<uint32_t>(16);

    // batch args
    uint32_t MtKt                               = get_arg_val<uint32_t>(17); // if 0
    uint32_t KtNt                               = get_arg_val<uint32_t>(18);
    uint32_t batch                              = get_arg_val<uint32_t>(19);
    uint32_t bcast_B                            = get_arg_val<uint32_t>(20);

    // WRITER
    // out tensor args
    uint32_t out_tensor_addr                    = get_arg_val<uint32_t>(21);
    uint32_t out_tensor_start_tile_id           = get_arg_val<uint32_t>(22);
    uint32_t out_tensor_stride_w                = get_arg_val<uint32_t>(23);
    uint32_t out_tensor_stride_h                = get_arg_val<uint32_t>(24);
    uint32_t out_tensor_next_subblock_stride_w  = get_arg_val<uint32_t>(25);
    uint32_t out_tensor_next_subblock_stride_h  = get_arg_val<uint32_t>(26);

    // out subblock args
    uint32_t out_subblock_w                     = get_arg_val<uint32_t>(27);
    uint32_t out_subblock_h                     = get_arg_val<uint32_t>(28);
    uint32_t out_subblock_tile_count            = get_arg_val<uint32_t>(29);
    uint32_t out_num_subblocks_w                = get_arg_val<uint32_t>(30);
    uint32_t out_num_subblocks_h                = get_arg_val<uint32_t>(31);

    // batch args
    uint32_t MtNt                               = get_arg_val<uint32_t>(32); // if 0
    // Don't need batch; same as batch from READER args

    // const args for tile-based bank-swizzled layout
    // could be added to the arg list in the future to test different
    // bank-swizzling configurations
    constexpr uint32_t num_used_dram_ch = 8;
    constexpr uint32_t num_used_dram_ch_pow2_exponent = 3;

    constexpr uint32_t cb_id_in1 = 1;

    // WRITER
    constexpr uint32_t cb_id_out0 = 16;

    uint32_t single_tile_size_bytes = get_tile_size(cb_id_in1);
    // uint32_t single_tile_size_bytes = get_tile_size(cb_id_out0); // Should be same

    uint32_t l1_write_addr_in1;

    #define tile_size_is_pow2 get_compile_time_arg_val(0) == 1
    #if (tile_size_is_pow2)
    constexpr uint32_t tile_size_pow2_exponent = get_compile_time_arg_val(1);
    const InterleavedPow2AddrGen<true> s1 = {
        .bank_base_address = in1_tensor_addr,


        .log_base_2_of_page_size = tile_size_pow2_exponent
    };
    const InterleavedPow2AddrGen<true> s = {
        .bank_base_address = out_tensor_addr,


        .log_base_2_of_page_size = tile_size_pow2_exponent // TODO(AP): refactor
    };
    #else
    const InterleavedAddrGen<true> s1 = {
        .bank_base_address = in1_tensor_addr,


        .page_size = single_tile_size_bytes
    };
    const InterleavedAddrGen<true> s = {
        .bank_base_address = out_tensor_addr,


        .page_size = single_tile_size_bytes
    };
    #endif

    for (uint32_t b = 0; b < batch; b++) {
        uint32_t in1_tensor_current_block_start_tile_id = in1_tensor_start_tile_id;
        for(uint32_t block = 0; block < num_blocks; block++) {
            cb_reserve_back(cb_id_in1, in1_block_num_tiles);

            l1_write_addr_in1 = get_write_ptr(cb_id_in1);

            uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_block_start_tile_id;
            for(uint32_t h = 0; h < in1_block_h; h++) {
                uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                for(uint32_t w = 0; w < in1_block_w; w++) {
                    uint64_t in1_tile_noc_addr = get_noc_addr(in1_tensor_tile_id, s1);
                    noc_async_read(in1_tile_noc_addr, l1_write_addr_in1, single_tile_size_bytes);
                    l1_write_addr_in1 += single_tile_size_bytes;
                    //DPRINT <<  h << " " << w << " " << block << ENDL();
                    in1_tensor_tile_id += in1_tensor_stride_w;
                }
                in1_tensor_row_start_tile_id += in1_tensor_stride_h;
            }
            //DPRINT << 1 << " " << block << ENDL();
            in1_tensor_current_block_start_tile_id += in1_tensor_next_block_stride;

            noc_async_read_barrier();

            cb_push_back(cb_id_in1, in1_block_num_tiles);

        }
        if (bcast_B == 0) {
            in1_tensor_start_tile_id += KtNt;
        }

        // WRITER
        uint32_t out_tensor_sbh_start_tile_id = out_tensor_start_tile_id;
        for(uint32_t sbh = 0; sbh < out_num_subblocks_h; sbh++) {
            uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
            for(uint32_t sbw = 0; sbw < out_num_subblocks_w; sbw++) {
                uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;

                cb_wait_front(cb_id_out0, out_subblock_tile_count);
                uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

                for(uint32_t h = 0; h < out_subblock_h; h++) {
                    uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                    for(uint32_t w = 0; w < out_subblock_w; w++) {
                        uint64_t out_tensor_tile_noc_addr = get_noc_addr(out_tensor_tile_id, s);

                        noc_async_write(l1_read_addr, out_tensor_tile_noc_addr, single_tile_size_bytes);
                        l1_read_addr+=single_tile_size_bytes;

                        out_tensor_tile_id += out_tensor_stride_w;
                    }
                    out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
                }

                noc_async_write_barrier();
                cb_pop_front(cb_id_out0, out_subblock_tile_count);
                out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
            }
            out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
        }
        out_tensor_start_tile_id += MtNt;
    }
}
