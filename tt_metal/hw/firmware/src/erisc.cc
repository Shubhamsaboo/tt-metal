// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "eth_l1_address_map.h"
#include "ethernet/dataflow_api.h"
#include "ethernet/tunneling.h"
#include "firmware_common.h"
#include "generated_bank_to_noc_coord_mapping.h"
#include "noc_parameters.h"
#include "risc_attribs.h"
#include "tools/profiler/kernel_profiler.hpp"

#ifdef __cplusplus
extern "C" {
#endif

void ApplicationHandler(void) __attribute__((__section__(".init")));

#ifdef __cplusplus
}
#endif

namespace kernel_profiler {
uint32_t wIndex __attribute__((used));
}

uint8_t noc_index = 0;  // TODO: remove hardcoding
uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));

void __attribute__((section("code_l1"))) risc_init() {
    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(n, 0, NOC_NODE_ID);
        my_x[n] = noc_id_reg & NOC_NODE_ID_MASK;
        my_y[n] = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    }
}

void __attribute__((section("erisc_l1_code"))) ApplicationHandler(void) {
    DEBUG_STATUS('I');
    rtos_context_switch_ptr = (void (*)())RtosTable[0];

    risc_init();
    noc_init();

    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        noc_local_state_init(n);
    }
    ncrisc_noc_full_sync();
    DEBUG_STATUS('R', 'E', 'W');
    while (routing_info->routing_enabled != 1) {
        internal_::risc_context_switch();
    }
    DEBUG_STATUS('R', 'E', 'D');


    while (routing_info->routing_enabled) {
        // FD: assume that no more host -> remote writes are pending
        if (erisc_info->launch_user_kernel == 1) {
            DEBUG_STATUS('R');
            kernel_profiler::init_profiler();
            kernel_profiler::mark_time(CC_MAIN_START);
            kernel_init();
            kernel_profiler::mark_time(CC_MAIN_END);
        } else {
            internal_::risc_context_switch();
        }
    }
    internal_::disable_erisc_app();
}
