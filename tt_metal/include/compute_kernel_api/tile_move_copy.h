// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"

#ifdef TRISC_MATH
#include "llk_math_unary_datacopy_api.h"
#endif

#ifdef TRISC_UNPACK
#include "llk_unpack_A_api.h"
#endif
namespace ckernel {

/**
 * Copies a single tile from the DST register buffer at a specified index to a
 * specified CB at a given index. For the out_tile_index to be valid for this
 * call, cb_reserve_back(n) had to be called first to reserve at least some
 * number n>0 of tiles in the output CB. The out_tile_index = 0 then references
 * the first tile in the reserved section of the CB, up to index n-1 that will
 * then be visible to the consumer in the same order after a cb_push_back call.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Each subsequent pack call will increment the write pointer in the cb by single
 * tile size. The pointer is then again set to a valid position with space for n
 * reserved tiles by another cb_reserve_back call.
 *
 * Operates in tandem with functions cb_reserve_back and cb_push_back.
 *
 * A typical use case is first the producer ensures that there is a number of
 * tiles available in the buffer via cb_reserve_back, then the producer uses
 * the pack_tile call to copy a tile from one of DST slots to a slot in
 * reserved space and finally cb_push_back is called to announce visibility of
 * the reserved section of the circular buffer to the consumer.
 *
 * Return value: None
 *
 * | Argument       | Description                                       | Type     | Valid Range                                         | Required |
 * |----------------|---------------------------------------------------|----------|-----------------------------------------------------|----------|
 * | ifrom_dst      | The index of the tile in the DST register         | uint32_t | Must be less than the size of the DST register (16) | True     |
 * | icb            | The identifier of the output circular buffer (CB) | uint32_t | 0 to 31                                             | True     |
 * | icb_tile       | The index of the tile in the output CB to copy to | uint32_t | Must be less than the size of the CB                | True     |
 */

/**
 * Perform the init short for copy tile. This does not reconfigure the unpacker data types.
 */
ALWI void copy_tile_to_dst_init_short(uint cbid = 0, uint transpose = 0)
{
    UNPACK(( llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE>(transpose)  ));
    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, DST_ACCUM_MODE>(false /*transpose of faces*/, false /*transpose within 16x16 face*/, cbid)  ));
}

/**
 * Perform a init for the copy tile operation. This calls the short init function and initializes packer dst offset registers.
 */
ALWI void copy_tile_init()
{
    copy_tile_to_dst_init_short();
    PACK(( llk_init_packer_dest_offset_registers<SyncHalf,DstTileFaceLayout::RowMajor,false>() ));
}

ALWI void copy_tile_to_dst_init_short_with_dt(uint32_t old_cbid, uint32_t new_cbid, uint transpose = 0) {
    // This reconfig call checks if old operand has different data format to
    // new operand idx, otherwise no reconfig call occurs
    UNPACK(( llk_unpack_reconfig_data_format_srca(old_cbid, new_cbid) ));
    copy_tile_to_dst_init_short(new_cbid, transpose);
}

ALWI void copy_tile_matmul_partials_init_short_with_dt(uint32_t old_cbid, uint32_t new_cbid) {
    #ifdef ARCH_GRAYSKULL
    copy_tile_to_dst_init_short_with_dt(old_cbid, new_cbid, true);
    #else
    copy_tile_to_dst_init_short_with_dt(old_cbid, new_cbid, false);
    #endif
}

/**
 * Copies a single tile from the specified input CB and writes the result to
 * DST at a specified index. For the in_tile_index to be valid for this call,
 * cb_wait_front(n) had to be previously called to ensure that at least some
 * number n>0 of tiles are available in the input CB. The CB index 0 then
 * references the first tile in the received section of the CB, up to index n-1
 * (in a FIFO order). The DST register buffer must be in acquired state via
 * acquire_dst call. This call is blocking and is only available on the compute
 * engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                       | Data type | Valid range                                         | required |
 * |----------------|---------------------------------------------------|-----------|-----------------------------------------------------|----------|
 * | in_cb_id       | The identifier of the source circular buffer (CB) | uint32_t  | 0 to 31                                             | Yes      |
 * | in_tile_index  | The index of the tile to copy from the input CB   | uint32_t  | Must be less than the size of the CB                | Yes      |
 * | dst_tile_index | The index of the tile in the DST register         | uint32_t  | Must be less than the size of the DST register (16) | Yes      |
 * */
ALWI void copy_tile(uint32_t icb, uint32_t itile, uint32_t idst)
{
    UNPACK(( llk_unpack_A<BroadcastType::NONE, false>(icb, itile)  ));
    MATH(( llk_math_eltwise_unary_datacopy<A2D, BroadcastType::NONE, SyncHalf, DST_ACCUM_MODE>(idst)  ));
}

ALWI void copy_block_matmul_partials(uint32_t icb, uint32_t start_itile, uint32_t start_idst, uint32_t ntiles)
{
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_A_block<BroadcastType::NONE, false>(icb, start_itile, ntiles, true)  ));
    #else
    UNPACK(( llk_unpack_A_block<BroadcastType::NONE, false>(icb, start_itile, ntiles, false)  ));
    #endif

    MATH(( llk_math_eltwise_unary_datacopy_block<A2D, BroadcastType::NONE, SyncHalf, DST_ACCUM_MODE>(start_idst, ntiles, icb)  ));
}

}
