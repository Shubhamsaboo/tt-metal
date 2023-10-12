// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "debug_print.h"

inline uint32_t generate_reader_pattern_indices(uint32_t image_height,
                                        uint32_t image_width,
                                        uint32_t stride_h,
                                        uint32_t stride_w,
                                        uint32_t img_flat_h_idx,
                                        volatile tt_l1_ptr std::uint32_t* reader_pattern,
                                        uint32_t reader_pattern_index,
                                        uint32_t top_partial_middle_aligned_row_width,
                                        uint32_t skip_top_partial_middle_aligned_row,
                                        uint32_t top_partial_right_aligned_row_width,
                                        uint32_t skip_top_partial_right_aligned_row,
                                        uint32_t num_rows_top_partial_image,
                                        uint32_t num_skip_rows_top_partial_image,
                                        uint32_t num_full_images,
                                        uint32_t num_rows_bottom_partial_image,
                                        uint32_t num_skip_rows_bottom_partial_image,
                                        uint32_t bottom_partial_left_aligned_row_width,
                                        uint32_t skip_bottom_partial_left_aligned_row) {

    if (!skip_top_partial_right_aligned_row) {
        for (uint32_t top_partial_row_width_i = 0; top_partial_row_width_i < top_partial_right_aligned_row_width; top_partial_row_width_i += stride_w) {
            reader_pattern[reader_pattern_index] = top_partial_row_width_i;
            reader_pattern_index += 1;
        }
    }

    img_flat_h_idx = top_partial_right_aligned_row_width;

    uint32_t img_flat_h_idx_top_partial_image = img_flat_h_idx +  (num_skip_rows_top_partial_image * image_width);
    for (uint32_t top_partial_image_row_i = num_skip_rows_top_partial_image; top_partial_image_row_i < num_rows_top_partial_image; top_partial_image_row_i += stride_h) {
        for (uint32_t row_width_i = 0; row_width_i < image_width; row_width_i += stride_w) {
            reader_pattern[reader_pattern_index] = img_flat_h_idx_top_partial_image + row_width_i;
            reader_pattern_index += 1;
        }
        img_flat_h_idx_top_partial_image += (stride_h * image_width);
    }

    img_flat_h_idx += (num_rows_top_partial_image * image_width);

    for (uint32_t full_img_i = 0; full_img_i < num_full_images; full_img_i++) {
        uint32_t img_flat_h_idx_curr_image = img_flat_h_idx;
        for (uint32_t img_row_idx = 0; img_row_idx < image_height; img_row_idx += stride_h) {
            for (uint32_t img_row_width_i = 0; img_row_width_i < image_width; img_row_width_i += stride_w) {
                reader_pattern[reader_pattern_index] = img_flat_h_idx_curr_image + img_row_width_i;
                reader_pattern_index += 1;
            }
            img_flat_h_idx_curr_image += (stride_h * image_width); // TODO: make them ct args.. no multiplies
        }
        img_flat_h_idx += (image_height * image_width);
    }

    uint32_t img_flat_h_idx_bottom_partial_image = img_flat_h_idx +  (num_skip_rows_bottom_partial_image * image_width);
    for (uint32_t bottom_partial_image_row_i = num_skip_rows_bottom_partial_image; bottom_partial_image_row_i < num_rows_bottom_partial_image; bottom_partial_image_row_i += stride_h) {
        for (uint32_t row_width_i = 0; row_width_i < image_width; row_width_i += stride_w) {
            reader_pattern[reader_pattern_index] = img_flat_h_idx_bottom_partial_image + row_width_i;
            reader_pattern_index += 1;
        }
        img_flat_h_idx_bottom_partial_image += (stride_h * image_width);
    }

    img_flat_h_idx += (num_rows_bottom_partial_image * image_width);

    if (!skip_bottom_partial_left_aligned_row) {
        for (uint32_t bottom_partial_row_width_i = 0; bottom_partial_row_width_i < bottom_partial_left_aligned_row_width; bottom_partial_row_width_i += stride_w) {
            reader_pattern[reader_pattern_index] = img_flat_h_idx + bottom_partial_row_width_i;
            reader_pattern_index += 1;
        }
    }
    return reader_pattern_index;
}

void kernel_main() {

    uint32_t i = 0;
    uint32_t image_height = get_arg_val<uint32_t>(i); i+=1;
    uint32_t image_width = get_arg_val<uint32_t>(i); i+=1;
    uint32_t stride_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t stride_w = get_arg_val<uint32_t>(i); i+=1;

    // args to grab halo data
    uint32_t halo_read_enabled = get_arg_val<uint32_t>(i); i+=1;
    uint32_t halo_core_noc_x = get_arg_val<uint32_t>(i); i+=1;
    uint32_t halo_core_noc_y = get_arg_val<uint32_t>(i); i+=1;
    uint32_t halo_num_tiles =  get_arg_val<uint32_t>(i); i+=1;
    uint32_t halo_start_addr =  get_arg_val<uint32_t>(i); i+=1;
    uint32_t halo_size_bytes =  get_arg_val<uint32_t>(i); i+=1;

    // halo region args
    uint32_t halo_read_pattern_offset = get_arg_val<uint32_t>(i); i+=1;
    uint32_t halo_top_partial_middle_aligned_row_width = get_arg_val<uint32_t>(i); i+=1;
    uint32_t halo_skip_top_partial_middle_right_aligned_row = get_arg_val<uint32_t>(i); i+=1;

    uint32_t halo_top_partial_right_aligned_row_width = get_arg_val<uint32_t>(i); i+=1;
    uint32_t halo_skip_top_partial_right_aligned_row = get_arg_val<uint32_t>(i); i+=1;

    uint32_t halo_num_rows_top_partial_image = get_arg_val<uint32_t>(i); i+=1;
    uint32_t halo_num_skip_rows_top_partial_image = get_arg_val<uint32_t>(i); i+=1;

    uint32_t halo_num_full_images = get_arg_val<uint32_t>(i); i+=1;

    uint32_t halo_num_rows_bottom_partial_image = get_arg_val<uint32_t>(i); i+=1;
    uint32_t halo_num_skip_rows_bottom_partial_image = get_arg_val<uint32_t>(i); i+=1;

    uint32_t halo_bottom_partial_left_aligned_row_width = get_arg_val<uint32_t>(i); i+=1;
    uint32_t halo_skip_bottom_partial_left_aligned_row = get_arg_val<uint32_t>(i); i+=1;

    // local region args
    uint32_t local_read_pattern_offset = get_arg_val<uint32_t>(i); i+=1;
    uint32_t local_top_partial_middle_aligned_row_width = get_arg_val<uint32_t>(i); i+=1;
    uint32_t local_skip_top_partial_middle_right_aligned_row = get_arg_val<uint32_t>(i); i+=1;

    uint32_t local_top_partial_right_aligned_row_width = get_arg_val<uint32_t>(i); i+=1;
    uint32_t local_skip_top_partial_right_aligned_row = get_arg_val<uint32_t>(i); i+=1;

    uint32_t local_num_rows_top_partial_image = get_arg_val<uint32_t>(i); i+=1;
    uint32_t local_num_skip_rows_top_partial_image = get_arg_val<uint32_t>(i); i+=1;

    uint32_t local_num_full_images = get_arg_val<uint32_t>(i); i+=1;

    uint32_t local_num_rows_bottom_partial_image = get_arg_val<uint32_t>(i); i+=1;
    uint32_t local_num_skip_rows_bottom_partial_image = get_arg_val<uint32_t>(i); i+=1;

    uint32_t local_bottom_partial_left_aligned_row_width = get_arg_val<uint32_t>(i); i+=1;
    uint32_t local_skip_bottom_partial_left_aligned_row = get_arg_val<uint32_t>(i); i+=1;

    // args are for untilizing and tilizing
    uint32_t num_untilized_input_blocks = get_arg_val<uint32_t>(i); i+=1; // input block contains only 1 row of tiles
    uint32_t num_tiles_untilized_input_block = get_arg_val<uint32_t>(i); i+=1;
    uint32_t num_output_tiles = get_arg_val<uint32_t>(i); i+=1;

    uint32_t noop = get_arg_val<uint32_t>(i); i+=1;
    if(noop) {
        return;
    }

    constexpr uint32_t untilize_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t untilize_downsampled_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t final_tilize_output_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t reader_pattern_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t df_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t conv_act_size_c_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t halo_input_cb_index = get_compile_time_arg_val(6);

    uint32_t img_flat_h_idx = 0;
    uint32_t reader_pattern_l1_addr = get_write_ptr(reader_pattern_cb_index);
    volatile tt_l1_ptr std::uint32_t* reader_pattern = (volatile tt_l1_ptr uint32_t*)(reader_pattern_l1_addr);
    uint32_t reader_pattern_index = 0;

    if (halo_read_enabled) {
        // Read # of row of tiles from previous core into local cb
        // noc_async_read_barrier
        // Push cb to compute for untilizing
        cb_reserve_back(halo_input_cb_index, halo_num_tiles);
        uint32_t halo_cb_write_addr = get_write_ptr(halo_input_cb_index);
        noc_async_read(get_noc_addr(halo_core_noc_x, halo_core_noc_y, halo_start_addr), halo_cb_write_addr, halo_size_bytes);
        noc_async_read_barrier();
        cb_push_back(halo_input_cb_index, halo_num_tiles);
        img_flat_h_idx = halo_read_pattern_offset;
        // generate reader pattern - halo region
        reader_pattern_index = generate_reader_pattern_indices(image_height,
                                                                image_width,
                                                                stride_h,
                                                                stride_w,
                                                                img_flat_h_idx,
                                                                reader_pattern,
                                                                reader_pattern_index,
                                                                halo_top_partial_middle_aligned_row_width,
                                                                halo_skip_top_partial_middle_right_aligned_row,
                                                                halo_top_partial_right_aligned_row_width,
                                                                halo_skip_top_partial_right_aligned_row,
                                                                halo_num_rows_top_partial_image,
                                                                halo_num_skip_rows_top_partial_image,
                                                                halo_num_full_images,
                                                                halo_num_rows_bottom_partial_image,
                                                                halo_num_skip_rows_bottom_partial_image,
                                                                halo_bottom_partial_left_aligned_row_width,
                                                                halo_skip_bottom_partial_left_aligned_row);
    }
    // halo and local data will be pushed to the same untilize cb
    // img flat h idx is index within the concatenated halo and local data
    // compute reader pattern - local region
    img_flat_h_idx = local_read_pattern_offset;
    reader_pattern_index = generate_reader_pattern_indices(image_height,
                                                            image_width,
                                                            stride_h,
                                                            stride_w,
                                                            img_flat_h_idx,
                                                            reader_pattern,
                                                            reader_pattern_index,
                                                            local_top_partial_middle_aligned_row_width,
                                                            local_skip_top_partial_middle_right_aligned_row,
                                                            local_top_partial_right_aligned_row_width,
                                                            local_skip_top_partial_right_aligned_row,
                                                            local_num_rows_top_partial_image,
                                                            local_num_skip_rows_top_partial_image,
                                                            local_num_full_images,
                                                            local_num_rows_bottom_partial_image,
                                                            local_num_skip_rows_bottom_partial_image,
                                                            local_bottom_partial_left_aligned_row_width,
                                                            local_skip_bottom_partial_left_aligned_row);

    // wait for untilized blocks from compute, drop rows based on reader_pattern and push to untilize_downsampled_cb_index
    reader_pattern_index = 0;
    uint32_t untilize_block_offset = 0;
    //DPRINT << "here" << ENDL();
    cb_reserve_back(untilize_downsampled_cb_index, num_output_tiles);
    //DPRINT << "reserved tiles in untilize downsampled cb" << ENDL();
    uint32_t untilize_downsampled_cb_l1_write_addr = get_write_ptr(untilize_downsampled_cb_index);
    uint32_t untilize_downsampled_cb_l1_write_addr_start = untilize_downsampled_cb_l1_write_addr;
    // num_untilized_input_blocks contains halo as well as local data
    for (uint32_t untilized_input_block_i = 0; untilized_input_block_i < num_untilized_input_blocks; untilized_input_block_i++) {
        //DPRINT << "going to wait for 1 row of tiles in untilized buffer, " << num_tiles_untilized_input_block << ENDL();
        cb_wait_front(untilize_cb_index, num_tiles_untilized_input_block); // 1 row of tiles
        //DPRINT << "waited for 1 row of tiles in untilized buffer, " << num_tiles_untilized_input_block << ENDL();
        uint32_t untilize_block_l1_read_addr = get_read_ptr(untilize_cb_index);
        // tt_l1_ptr BF16* untilize_block_array = (tt_l1_ptr BF16*)(untilize_block_l1_read_addr);
        // if (untilized_input_block_i == 0) {
        // for(uint32_t i = 0; i < 128; i += 1) {
        //     DPRINT << "untilize_block_array[" << i << "]=" << BF16(untilize_block_array[i]) << ENDL();
        // }
        // }
        while(reader_pattern[reader_pattern_index] < 32 + untilize_block_offset) {
            // local read into untilized downsampled l1, read_address = l1_read_addr + (reader_pattern[reader_pattern_index] * conv_act_c_bytes), read_size = conv_act_c_bytes, write_address = untilize_downsampled_cb_l1_write_addr
            //DPRINT << "reader_pattern[reader_pattern_index]=" << reader_pattern[reader_pattern_index] << ENDL();
            //qDPRINT << "conv_act_size_c_bytes=" << conv_act_size_c_bytes << ENDL();
            uint32_t read_address = untilize_block_l1_read_addr + ((reader_pattern[reader_pattern_index]-untilize_block_offset)  * conv_act_size_c_bytes);
            noc_async_read(get_noc_addr(read_address), untilize_downsampled_cb_l1_write_addr, conv_act_size_c_bytes);
            untilize_downsampled_cb_l1_write_addr += conv_act_size_c_bytes;
            reader_pattern_index += 1;
        }
        //DPRINT << "waiting noc barrier" << ENDL();
        noc_async_read_barrier();
        //DPRINT << "downsampled 1 row of untilized cb buffer" << ENDL();
        cb_pop_front(untilize_cb_index, num_tiles_untilized_input_block);
        untilize_block_offset += 32;
    }
    // tt_l1_ptr BF16* untilize_ds_array = (tt_l1_ptr BF16*)(untilize_downsampled_cb_l1_write_addr_start);
    // // for(uint32_t i = 0; i < 64; i += 1) {
    // //     DPRINT << "untilize_ds_array[" << i << "]=" << BF16(untilize_ds_array[i]) << ENDL();
    // // }

    cb_push_back(untilize_downsampled_cb_index, num_output_tiles); // to be tilized by compute kernel
    //DPRINT << "Pushed untilize downsampled cb" << ENDL();
    // wait for tilized cb
    cb_wait_front(final_tilize_output_cb_index, num_output_tiles); // wait we dont need this.. sharded output.
}
