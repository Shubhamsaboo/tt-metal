/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <mutex>
#include <variant>

#include "third_party/magic_enum/magic_enum.hpp"

#include "tt_metal/build_kernels_for_riscv/build_kernels_for_riscv.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/detail/program.hpp"
#include "tt_metal/llrt/watcher.hpp"
#include "tt_metal/third_party/umd/device/util.hpp"

using std::unique_lock;
using std::mutex;

namespace tt::tt_metal{

    namespace detail {
        // To be removed at a later time, but need a global
        // command queue for the time being.
        inline unique_ptr<CommandQueue> GLOBAL_CQ;

        /**
         * Read device side profiler data and dump results into device side CSV log
         *
         * Return value: void
         *
         * | Argument      | Description                                       | Type            | Valid Range               | Required |
         * |---------------|---------------------------------------------------|-----------------|---------------------------|----------|
         * | device        | The device holding the program being profiled.    | Device *        |                           | True     |
         * | program       | The program being profiled.                       | const Program & |                           | True     |
         * */
        void DumpDeviceProfileResults(Device *device, const Program &program);

        /**
         * Set the directory for all CSV logs produced by the profiler instance in the tt-metal module
         *
         * Return value: void
         *
         * | Argument     | Description                                             |  Data type  | Valid range              | required |
         * |--------------|---------------------------------------------------------|-------------|--------------------------|----------|
         * | output_dir   | The output directory that will hold the outpu CSV logs  | std::string | Any valid directory path | No       |
         * */
        void SetProfilerDir(std::string output_dir = "");

        /**
         * Start a fresh log for the host side profile results
         *
         * Return value: void
         *
         * | Argument     | Description                                             |  Data type  | Valid range              | required |
         * |--------------|---------------------------------------------------------|-------------|--------------------------|----------|
         * */
        void FreshProfilerHostLog();

        /**
         * Start a fresh log for the device side profile results
         *
         * Return value: void
         *
         * | Argument     | Description                                             |  Data type  | Valid range              | required |
         * |--------------|---------------------------------------------------------|-------------|--------------------------|----------|
         * */
        void FreshProfilerDeviceLog();

        /**
         * Profile scopes in tt_metal API
         *
         * */

        class ProfileTTMetalScope
        {
            private:
                string scopeName = "";
            public:
                ProfileTTMetalScope (const string& scopeNameArg);
                ~ProfileTTMetalScope ();
        };

        /**
         * Copies data from a host buffer into a buffer within the device DRAM channel
         *
         * Return value: bool
         *
         * | Argument     | Description                                            | Data type             | Valid range                               | required |
         * |--------------|--------------------------------------------------------|-----------------------|-------------------------------------------|----------|
         * | device       | The device whose DRAM to write data into               | Device *              |                                           | Yes      |
         * | dram_channel | Channel index of DRAM to write into                    | int                   | On Grayskull, [0, 7] inclusive            | Yes      |
         * | address      | Starting address on DRAM channel to begin writing data | uint32_t              |                                           | Yes      |
         * | host_buffer  | Buffer on host to copy data from                       | std::vector<uint32_t> | Host buffer must be fully fit DRAM buffer | Yes      |
         */
        inline bool WriteToDeviceDRAMChannel(Device *device, int dram_channel, uint32_t address, std::vector<uint32_t> &host_buffer)
        {
            bool pass = true;
            device->cluster()->write_dram_vec(host_buffer, tt_target_dram{device->pcie_slot(), dram_channel, 0}, address);
            return pass;
        }

        /**
         * Copy data from a device DRAM channel to a host buffer
         *
         * Return value: bool
         *
         * | Argument     | Description                                                  | Data type             | Valid range                    | required |
         * |--------------|--------------------------------------------------------------|-----------------------|--------------------------------|----------|
         * | device       | The device whose DRAM to read data from                      | Device *              |                                | Yes      |
         * | dram_channel | Channel index of DRAM to read from                           | int                   | On Grayskull, [0, 7] inclusive | Yes      |
         * | address      | Starting address on DRAM channel from which to begin reading | uint32_t              |                                | Yes      |
         * | size         | Size of buffer to read from device in bytes                  | uint32_t              |                                | Yes      |
         * | host_buffer  | Buffer on host to copy data into                             | std::vector<uint32_t> |                                | Yes      |
         */
        inline bool ReadFromDeviceDRAMChannel(Device *device, int dram_channel, uint32_t address, uint32_t size, std::vector<uint32_t> &host_buffer)
        {
            bool pass = true;
            device->cluster()->read_dram_vec(host_buffer, tt_target_dram{device->pcie_slot(), dram_channel, 0}, address, size);
            return pass;
        }

        /**
         * Copy data from a host buffer into an L1 buffer. (Note: Current Can not be a CircularBuffer.)
         *
         * Return value: bool
         *
         * | Argument      | Description                                     | Data type             | Valid range                                         | required |
         * |---------------|-------------------------------------------------|-----------------------|-----------------------------------------------------|----------|
         * | device        | The device whose DRAM to write data into        | Device *              |                                                     | Yes      |
         * | logical_core  | Logical coordinate of core whose L1 to write to | CoreCoord            | On Grayskull, any valid logical worker coordinate   | Yes      |
         * | address       | Starting address in L1 to write into            | uint32_t              | Any non-reserved address in L1 that fits for buffer | Yes      |
         * | host_buffer   | Buffer on host whose data to copy from          | std::vector<uint32_t> | Buffer must fit into L1                             | Yes      |
         */
        inline bool WriteToDeviceL1(Device *device, const CoreCoord &logical_core, uint32_t address, std::vector<uint32_t> &host_buffer)
        {
            auto worker_core = device->worker_core_from_logical_core(logical_core);
            llrt::write_hex_vec_to_core(device->cluster(), device->pcie_slot(), worker_core, host_buffer, address);
            return true;
        }

        inline bool WriteToDeviceL1(Device *device, const CoreCoord &core, op_info_t op_info, int op_idx)
        {
            auto worker_core = device->worker_core_from_logical_core(core);
            llrt::write_graph_interpreter_op_info_to_core(device->cluster(), device->pcie_slot(), worker_core, op_info, op_idx);
            return true;
        }


        /**
         * Copy data from an L1 buffer into a host buffer. Must be a buffer, and not a CB.
         *
         * Return value: bool
         *
         * | Argument             | Description                                 | Data type             | Valid range                                       | required |
         * |----------------------|---------------------------------------------|-----------------------|---------------------------------------------------|----------|
         * | device               | The device whose DRAM to read data from     | Device *              |                                                   | Yes      |
         * | logical_core         | Logical coordinate of core whose L1 to read | CoreCoord            | On Grayskull, any valid logical worker coordinate | Yes      |
         * | address              | Starting address in L1 to read from         | uint32_t              |                                                   | Yes      |
         * | size                 | Size of L1 buffer in bytes                  | uint32_t              |                                                   | Yes      |
         * | host_buffer          | Buffer on host to copy data into            | std::vector<uint32_t> | Buffer must fit L1 buffer                         | Yes      |
         */
        inline bool ReadFromDeviceL1(Device *device, const CoreCoord &logical_core, uint32_t address, uint32_t size, std::vector<uint32_t> &host_buffer)
        {
            auto worker_core = device->worker_core_from_logical_core(logical_core);
            host_buffer = llrt::read_hex_vec_from_core(device->cluster(), device->pcie_slot(), worker_core, address, size);
            return true;
        }

        inline void Synchronize()
        {
            if (detail::GLOBAL_CQ) {
                Finish(*detail::GLOBAL_CQ);
            }
        }

        inline void GenerateDeviceHeaders(Device *device,
                                          build_kernel_for_riscv_options_t *build_options,
                                          const std::string &op_path_suffix)
        {
            // Basic Allocator generates number of banks which may not be power of 2, so we could just pad and alias for now
            const size_t num_dram_banks = device->num_banks(BufferType::DRAM);
            const size_t num_dram_banks_pow2 = std::pow(2, std::ceil(std::log2(num_dram_banks)));
            std::vector<CoreCoord> dram_noc_coord_per_bank(num_dram_banks);
            std::vector<i32> dram_offsets_per_bank(num_dram_banks);
            for (unsigned bank_id = 0; bank_id < num_dram_banks; bank_id++) {
                dram_noc_coord_per_bank[bank_id] = device->core_from_dram_channel(device->dram_channel_from_bank_id(bank_id));
                dram_offsets_per_bank[bank_id] = device->dram_bank_offset_from_bank_id(bank_id);
            }
            const size_t num_l1_banks = device->num_banks(BufferType::L1);
            const size_t num_l1_banks_pow2 = std::pow(2, std::ceil(std::log2(num_l1_banks)));
            std::vector<CoreCoord> l1_noc_coord_per_bank(num_l1_banks_pow2);
            std::vector<i32> l1_offset_per_bank(num_l1_banks_pow2);
            for (unsigned bank_id = 0; bank_id < num_l1_banks_pow2; bank_id++) {
                if (bank_id < num_l1_banks) {
                    l1_noc_coord_per_bank[bank_id] = device->worker_core_from_logical_core(device->logical_core_from_bank_id(bank_id));
                    l1_offset_per_bank[bank_id] = device->l1_bank_offset_from_bank_id(bank_id);
                } else {
                    l1_noc_coord_per_bank[bank_id] = device->worker_core_from_logical_core(device->logical_core_from_bank_id(0));
                    l1_offset_per_bank[bank_id] = device->l1_bank_offset_from_bank_id(0);
                }
            }
            // Generate header file in proper location
            generate_bank_to_noc_coord_descriptor (
                build_options,
                op_path_suffix,
                dram_noc_coord_per_bank,
                dram_offsets_per_bank,
                l1_noc_coord_per_bank,
                l1_offset_per_bank
            );

            tt_SocDescriptor& soc_d = device->cluster()->get_soc_desc(device->pcie_slot());

            // Determine which noc-coords are harvested
            // TODO(PGK/Almeet): fix this w/ new UMD
            vector<uint32_t> harvested;
            uint32_t harvested_noc_rows = device->cluster()->get_harvested_rows(device->pcie_slot());
            for (uint32_t y = 0; y < soc_d.grid_size.y; y++) {
                bool row_harvested = (harvested_noc_rows >> y) & 0x1;
                if (row_harvested) {
                    harvested.push_back(y);
                }
            }

            // XXXX TODO(PGK): get addr range values from device descriptor...
            generate_noc_addr_ranges_header (
                build_options,
                op_path_suffix,
                0, (uint64_t)4 * 1024 * 1024 * 1024,
                0, 1 * 1024 * 1024 * 1024,
                soc_d.get_pcie_cores(),
                soc_d.get_dram_cores(),
                soc_d.get_ethernet_cores(),
                soc_d.grid_size,
                harvested);
        }

        inline DataMovementConfig GetDataMovementConfig(const Program &program, const std::string &file_name, const CoreRangeSet &core_ranges, const std::optional<DataMovementConfig> &dm_config) {
            bool riscv0_in_use = false; bool riscv1_in_use = false;
            bool noc0_in_use = false; bool noc1_in_use = false;

            auto set_global_and_local_noc_usage = [&](KernelID kernel_id, bool &local_noc0_usage, bool &local_noc1_usage) {
                const auto kernel = detail::GetKernel(program, kernel_id);
                auto kernel_config = std::get<DataMovementConfig>(kernel->config());
                auto noc_value = magic_enum::enum_integer(kernel_config.noc);
                noc0_in_use, local_noc0_usage = noc_value == 0;
                noc1_in_use, local_noc1_usage = noc_value == 1;
            };

            for (const auto &core_range : core_ranges.ranges()) {
                for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
                    for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
                        auto kernel_group = program.kernels_on_core(CoreCoord(x, y));
                        bool local_noc0_in_use = false; bool local_noc1_in_use = false;
                        if (kernel_group.riscv0_id.has_value()) {
                            riscv0_in_use = true;
                            set_global_and_local_noc_usage(kernel_group.riscv0_id.value(), local_noc0_in_use, local_noc1_in_use);
                        }
                        if (kernel_group.riscv1_id.has_value()) {
                            riscv1_in_use = true;
                            set_global_and_local_noc_usage(kernel_group.riscv1_id.value(), local_noc0_in_use, local_noc1_in_use);
                        }
                        if (kernel_group.riscv0_id.has_value() and kernel_group.riscv1_id.has_value()) {
                            TT_ASSERT(local_noc0_in_use and local_noc1_in_use, "Illegal NOC usage: data movement kernels on logical core {} cannot use the same NOC, doing so results in hangs!");
                        }
                    }
                }
            }

            TT_ASSERT(not (riscv0_in_use and riscv1_in_use), "DataMovementKernel creation failure: Cannot create data movement kernel for {} across specified cores because both data movement processors are in use!", file_name);
            TT_ASSERT(not (noc0_in_use and noc1_in_use), "DataMovementKernel creation failure: Cannot create data movement kernels for {} across specified cores because both NOCs are in use!", file_name);

            if (dm_config.has_value()) {
                return dm_config.value();
            }

            DataMovementProcessor processor = riscv0_in_use ? DataMovementProcessor::RISCV_1 : DataMovementProcessor::RISCV_0;
            NOC noc = noc0_in_use ? NOC::NOC_1 : NOC::NOC_0;
            return DataMovementConfig{.processor = processor, .noc = noc};
        }

        inline CoreRangeSet GetCoreRangeSet(const std::variant<CoreCoord, CoreRange, CoreRangeSet> &specified_core_spec) {
            return std::visit(
                [](auto&& core_spec) -> CoreRangeSet
                {
                    using T = std::decay_t<decltype(core_spec)>;
                    if constexpr (std::is_same_v<T, CoreCoord>) {
                        return CoreRangeSet({CoreRange{.start=core_spec, .end=core_spec}});
                    }
                    else if constexpr (std::is_same_v<T, CoreRange>) {
                        return CoreRangeSet({core_spec});
                    }
                    else if constexpr (std::is_same_v<T, CoreRangeSet>) {
                        return core_spec;
                    }
                },
                specified_core_spec
            );
        }

        // TODO (abhullar): Remove this when tt_cluster and tt_metal::Device abstractions are redesigned
        class ClusterWrapper {
           public:
            ClusterWrapper& operator=(const ClusterWrapper&) = delete;
            ClusterWrapper& operator=(ClusterWrapper&& other) noexcept = delete;
            ClusterWrapper(const ClusterWrapper&) = delete;
            ClusterWrapper(ClusterWrapper&& other) noexcept = delete;

            static const ClusterWrapper& inst(const tt::ARCH &arch, const TargetDevice &target_type) {
                static ClusterWrapper inst(arch, target_type);
                return inst;
            }

            tt_cluster *cluster() const { return this->cluster_.get(); }

           private:
            ClusterWrapper(const tt::ARCH &arch, const TargetDevice &target_type) {
                ZoneScoped;
                this->cluster_ = std::make_unique<tt_cluster>();

                std::vector<chip_id_t> avail_device_ids = tt_SiliconDevice::detect_available_device_ids(true, false);
                std::set<chip_id_t> device_ids(avail_device_ids.begin(), avail_device_ids.end());

                const std::string sdesc_file = get_soc_description_file(arch, target_type);
                const std::string ndesc_path = (arch == tt::ARCH::WORMHOLE_B0) ? GetClusterDescYAML().string() : "";

                // init UMD with all available device IDs
                this->cluster_->open_device(arch, target_type, device_ids, sdesc_file, ndesc_path);

                tt_device_params default_params;
                if (getenv("TT_METAL_VERSIM_DUMP_CORES")) {
                    std::string dump_cores_string = getenv("TT_METAL_VERSIM_DUMP_CORES");
                    default_params.vcd_dump_cores = tt::utils::strsplit(dump_cores_string, ',');
                }

                this->cluster_->start_device(default_params);
            }
            ~ClusterWrapper() {
                log_info(tt::LogMetal, "Closing device driver");
                this->cluster_->close_device();
            }

            std::unique_ptr<tt_cluster> cluster_ = nullptr;
        };
    }
}
