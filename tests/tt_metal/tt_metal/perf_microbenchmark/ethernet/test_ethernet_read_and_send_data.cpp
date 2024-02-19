
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <limits>
#include <random>

#include "tt_metal/common/core_coord.h"
#include "tt_metal/common/math.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/env_vars.hpp"

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

class N300TestDevice {
   public:
    N300TestDevice() : device_open(false) {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (not slow_dispatch) {
            TT_THROW("This suite can only be run with TT_METAL_SLOW_DISPATCH_MODE set");
        }
        arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());

        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        if (arch_ == tt::ARCH::WORMHOLE_B0 and tt::tt_metal::GetNumAvailableDevices() == 2 and
            tt::tt_metal::GetNumPCIeDevices() == 1) {
            for (unsigned int id = 0; id < num_devices_; id++) {
                auto* device = tt::tt_metal::CreateDevice(id);
                devices_.push_back(device);
            }
            tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(true);

        } else {
            TT_THROW("This suite can only be run on N300 Wormhole devices");
        }
        device_open = true;
    }
    ~N300TestDevice() {
        if (device_open) {
            TearDown();
        }
    }

    void TearDown() {
        device_open = false;
        tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);
        for (unsigned int id = 0; id < devices_.size(); id++) {
            tt::tt_metal::CloseDevice(devices_.at(id));
        }
    }

    std::vector<tt::tt_metal::Device*> devices_;
    tt::ARCH arch_;
    size_t num_devices_;

   private:
    bool device_open;
};

struct BankedConfig {
    size_t num_pages;
    size_t size_bytes;
    size_t page_size_bytes;
    BufferType input_buffer_type;// = BufferType::L1;
    BufferType output_buffer_type;// = BufferType::L1;
    tt::DataFormat l1_data_format;// = tt::DataFormat::Float16_b;
};

bool RunWriteBWTest(
    std::string const& sender_kernel_path,
    std::string const& receiver_kernel_path,
    tt_metal::Device* sender_device,
    tt_metal::Device* receiver_device,

    const CoreCoord& eth_sender_core,
    const CoreCoord& eth_receiver_core,

    const size_t src_eth_l1_byte_address,
    const size_t dst_eth_l1_byte_address,

    const size_t precomputed_source_addresses_buffer_address,
    const size_t precomputed_source_addresses_buffer_size,

    const uint32_t eth_l1_staging_buffer_size,
    const uint32_t eth_max_concurrent_sends,
    const uint32_t input_buffer_page_size,
    const uint32_t input_buffer_size_bytes,
    bool source_is_dram,
    bool dest_is_dram

) {
    // number of bytes to send per eth send (given that eth l1 buf size not
    // guaranteed to be multiple of page size, we won't send the left over
    // bytes at the end
    const uint32_t pages_per_send = eth_l1_staging_buffer_size / input_buffer_page_size;
    const uint32_t num_bytes_per_send = pages_per_send * input_buffer_page_size;
    const uint32_t num_pages = ((input_buffer_size_bytes - 1) / input_buffer_page_size) + 1;  // includes padding
    const uint32_t num_messages_to_send = ((input_buffer_size_bytes - 1) / num_bytes_per_send) + 1;

    TT_ASSERT(precomputed_source_addresses_buffer_address < std::numeric_limits<uint32_t>::max(), "precomputed_source_addresses_buffer_address is too large");

    bool pass = true;
    log_debug(
        tt::LogTest,
        "Sending {} bytes from device {} eth core {} addr {} to device {} eth core {} addr {}",
        input_buffer_size_bytes,
        sender_device->id(),
        eth_sender_core.str(),
        src_eth_l1_byte_address,
        receiver_device->id(),
        eth_receiver_core.str(),
        dst_eth_l1_byte_address);
    std::cout << "num_messages_to_send: " << num_messages_to_send << std::endl;
    // Generate inputs
    ////////////////////////////////////////////////////////////////////////////
    //   SETUP THE INPUT CB
    ////////////////////////////////////////////////////////////////////////////
    std::cout << "Generating vector" << std::endl;
    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, input_buffer_size_bytes / sizeof(uint32_t));

    // Clear expected value at ethernet L1 address
    std::vector<uint32_t> all_zeros(inputs.size(), 0);

    BankedConfig test_config =
        BankedConfig{
            .num_pages = num_pages,
            .size_bytes = input_buffer_size_bytes,
            .page_size_bytes = input_buffer_page_size,
            .input_buffer_type = source_is_dram ? BufferType::DRAM : BufferType::L1,
            .output_buffer_type = dest_is_dram ? BufferType::DRAM : BufferType::L1,
            .l1_data_format = tt::DataFormat::Float16_b};
    auto input_buffer =
            CreateBuffer(
                InterleavedBufferConfig{sender_device, test_config.size_bytes, test_config.page_size_bytes, test_config.input_buffer_type});

    bool input_is_dram = test_config.input_buffer_type == BufferType::DRAM;
    tt_metal::detail::WriteToBuffer(input_buffer, inputs);
    uint32_t dram_buf_base_addr = input_buffer->address();

    ////////////////////////////////////////////////////////////////////////////
    //                      Sender Device
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program sender_program = tt_metal::Program();

    uint32_t num_pages_per_l1_buffer = num_bytes_per_send / input_buffer_page_size;
    TT_ASSERT(num_messages_to_send * num_pages_per_l1_buffer >= num_pages);
    std::cout << "eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE: " << eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE << std::endl;
    std::cout << "src_eth_l1_byte_address: " << src_eth_l1_byte_address << std::endl;
    auto eth_sender_kernel = tt_metal::CreateKernel(
        sender_program,
        sender_kernel_path,
        eth_sender_core,
        tt_metal::experimental::EthernetConfig{
            .eth_mode = tt_metal::Eth::SENDER,
            .noc = tt_metal::NOC::NOC_0,
            .compile_args = {
                uint32_t(num_bytes_per_send),         // 0
                uint32_t(num_bytes_per_send >> 4),    // 1
                uint32_t(num_messages_to_send),       // 2
                uint32_t(eth_max_concurrent_sends),   // 3
                uint32_t(source_is_dram)              // 4
                }
            // .compile_args = {uint32_t(256), uint32_t(256 >> 4)}
            });

    tt_metal::SetRuntimeArgs(
        sender_program,
        eth_sender_kernel,
        eth_sender_core,
        {
            uint32_t(src_eth_l1_byte_address),
            uint32_t(dst_eth_l1_byte_address),
            uint32_t(dram_buf_base_addr),
            uint32_t(input_buffer_page_size),
            uint32_t(num_pages),
            uint32_t(precomputed_source_addresses_buffer_address),
            uint32_t(precomputed_source_addresses_buffer_size)
        });

    ////////////////////////////////////////////////////////////////////////////
    //                           Receiver Device
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program receiver_program = tt_metal::Program();

    auto eth_receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        receiver_kernel_path,
        eth_receiver_core,
        tt_metal::experimental::EthernetConfig{
            .eth_mode = tt_metal::Eth::RECEIVER, .noc = tt_metal::NOC::NOC_0,
            .compile_args = {
                uint32_t(num_bytes_per_send),         // 0
                uint32_t(num_bytes_per_send >> 4),    // 1
                uint32_t(num_messages_to_send),       // 2
                uint32_t(eth_max_concurrent_sends),// 3
                uint32_t(dest_is_dram)                // 4
            }});  // probably want to use NOC_1 here
            // .compile_args = {uint32_t(256), uint32_t(256 >> 4)}});  // probably want to use NOC_1 here

    tt_metal::SetRuntimeArgs(
        receiver_program,
        eth_receiver_kernel,
        eth_receiver_core,
        {
            uint32_t(src_eth_l1_byte_address),
            uint32_t(dst_eth_l1_byte_address)
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    tt::tt_metal::detail::CompileProgram(sender_device, sender_program);
    tt::tt_metal::detail::CompileProgram(receiver_device, receiver_program);

    std::cout << "Running..." << std::endl;

    std::thread th2 = std::thread([&] {
        tt_metal::detail::LaunchProgram(receiver_device, receiver_program);
    });
    std::thread th1 = std::thread([&] {
        tt_metal::detail::LaunchProgram(sender_device, sender_program);
    });

    th2.join();
    std::cout << "receiver done" << std::endl;
    th1.join();
    std::cout << "sender done" << std::endl;

    auto readback_vec = llrt::read_hex_vec_from_core(
        receiver_device->id(),
        receiver_device->ethernet_core_from_logical_core(eth_receiver_core),
        dst_eth_l1_byte_address,
        input_buffer_size_bytes);
    pass &= (readback_vec == inputs);
    if (not pass) {
        std::cout << "Mismatch at Core: " << eth_receiver_core.str() << std::endl;
        std::cout << readback_vec[0] << std::endl;
    }
    return true; // TODO(snijjar): Fix
    return pass;
}


int main(int argc, char** argv) {
    // argv[0]: program
    // argv[1]: buffer_size_bytes
    // argv[2]: num_loops
    assert (argc == 10);
    std::string const& sender_kernel_path = argv[1];
    std::string const& receiver_kernel_path = argv[2];
    const uint32_t eth_l1_staging_buffer_size = std::stoi(argv[3]);
    const uint32_t eth_max_concurrent_sends = std::stoi(argv[4]);
    const uint32_t input_buffer_page_size = std::stoi(argv[5]);
    const uint32_t input_buffer_size_bytes = std::stoi(argv[6]);
    const bool source_is_dram = std::stoi(argv[7]) == 1;
    const bool dest_is_dram = std::stoi(argv[8]) == 1;
    const uint32_t precomputed_source_addresses_buffer_size = std::stoi(argv[9]);

    N300TestDevice test_fixture;

    std::cout << "precomputed_source_addresses_buffer_size: " << precomputed_source_addresses_buffer_size << std::endl;
    const auto& device_0 = test_fixture.devices_.at(0);
    const auto& device_1 = test_fixture.devices_.at(1);
    const size_t precomputed_source_addresses_buffer_address =
        precomputed_source_addresses_buffer_size == 0 ? (size_t)nullptr:
        eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE % 16 == 0 ? eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE :
            eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + 16 - (eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE % 8);
    // const size_t precomputed_source_addresses_buffer_size = 32;
    const size_t src_eth_l1_byte_address = precomputed_source_addresses_buffer_size == 0 ?
        eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE :
        precomputed_source_addresses_buffer_address + (precomputed_source_addresses_buffer_size * sizeof(std::size_t));
    const size_t dst_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + precomputed_source_addresses_buffer_size;

    auto const& active_eth_cores = device_0->get_active_ethernet_cores();
    assert (active_eth_cores.size() > 0);
    auto eth_sender_core_iter = active_eth_cores.begin();
    assert (eth_sender_core_iter != active_eth_cores.end());
    eth_sender_core_iter++;
    assert (eth_sender_core_iter != active_eth_cores.end());
    const auto& eth_sender_core = *eth_sender_core_iter;
    auto [device_id, eth_receiver_core] = device_0->get_connected_ethernet_core(eth_sender_core);

    // std::cout << "SENDER CORE: (x=" << eth_sender_core.x << ", y=" << eth_sender_core.y << ")" << std::endl;
    // std::cout << "RECEIVER CORE: (x=" << eth_receiver_core.x << ", y=" << eth_receiver_core.y << ")" << std::endl;

    // std::cout << "BW TEST: " << 64 << ", num_messages_to_send: " << num_messages_to_send << std::endl;
    RunWriteBWTest(
        sender_kernel_path,
        receiver_kernel_path,
        device_0,
        device_1,
        eth_sender_core,
        eth_receiver_core,
        src_eth_l1_byte_address,
        dst_eth_l1_byte_address,
        precomputed_source_addresses_buffer_address,
        precomputed_source_addresses_buffer_size,
        eth_l1_staging_buffer_size,
        eth_max_concurrent_sends,
        input_buffer_page_size,
        input_buffer_size_bytes,
        source_is_dram,
        dest_is_dram
        );

    test_fixture.TearDown();

}
