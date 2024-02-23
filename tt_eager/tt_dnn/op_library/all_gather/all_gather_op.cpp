// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/all_gather/all_gather_op.hpp"
#include "tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

#include "eth_l1_address_map.h"
#include "tt_metal/tools/profiler/op_profiler.hpp"

namespace tt {

namespace tt_metal {

void AllGather::validate(const std::vector<Tensor> &input_tensors) const {
    TT_FATAL(input_tensors.size() == 1);
    const auto& input_tensor = input_tensors[0];
    const auto& layout = input_tensor.layout();
    const auto& dtype = input_tensor.dtype();
    const auto& page_size = input_tensor.buffer()->page_size();
    TT_FATAL(page_size <= all_gather_buffer_params::eth_buffer_size, "Page size too large");
    // TODO: This can be removed by passing two page sizes, actual and aligned to be used for address offsets
    // Buffer sizes also need to take this aligned page size into consideration
    TT_FATAL(page_size % 32 == 0, "All Gather currently requires aligned pages");
    // TODO: Validate ring
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_gather need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr , "Operands to all_gather need to be allocated in buffers on device!");
    TT_FATAL(this->num_links > 0);
    TT_FATAL(this->num_links <= input_tensor.device()->compute_with_storage_grid_size().y, "Worker cores used by links are parallelizaed over rows");
    TT_FATAL(all_gather_buffer_params::num_buffers <= input_tensor.device()->compute_with_storage_grid_size().x, "Worker cores used by eth buffers are parallelizaed over cols");
    if (this->receiver_device_id == this->sender_device_id) {
        TT_FATAL(input_tensor.device()->get_ethernet_sockets(this->receiver_device_id).size() >= 2 * this->num_links, "2 Device all gather requires at least 2 eth connections per link");
    } else {
        TT_FATAL(input_tensor.device()->get_ethernet_sockets(this->receiver_device_id).size() >= this->num_links, "All gather requires at least 1 eth connection per link between sender and receiver device");
        TT_FATAL(input_tensor.device()->get_ethernet_sockets(this->sender_device_id).size() >= this->num_links, "All gather requires at least 1 eth connection per link between sender and receiver device");
    }
    if (input_tensor.is_sharded()) {
        // TODO: Kernels should already support concatting shards on width or height
        // We just need to take in a param to handle this
        // Currently width/block sharding will concat on width, height sharding will concat on height
        TT_FATAL(this->output_mem_config.memory_layout == input_tensor.memory_config().memory_layout);
        if (this->output_mem_config.shard_spec.has_value()) {
            const auto input_shard_spec = input_tensor.shard_spec().value();
            const auto output_shard_spec = this->output_mem_config.shard_spec.value();
            TT_FATAL(input_shard_spec.grid == output_shard_spec.grid);
            TT_FATAL(input_shard_spec.orientation == output_shard_spec.orientation);
            if (input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
                TT_FATAL(input_shard_spec.shape[0] * this->ring_size == output_shard_spec.shape[0]);
                TT_FATAL(input_shard_spec.shape[1] == output_shard_spec.shape[1]);
            } else {
                TT_FATAL(input_shard_spec.shape[0] == output_shard_spec.shape[0]);
                TT_FATAL(input_shard_spec.shape[1] * this->ring_size == output_shard_spec.shape[1]);
            }
        }
    } else {
        TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
    }
}

std::vector<Shape> AllGather::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    auto shape = input_tensors[0].shape();
    shape[this->dim] *= this->ring_size;
    return std::vector<Shape>(input_tensors.size(), shape);
}

std::vector<Tensor> AllGather::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    if (this->output_mem_config.is_sharded()) {
        auto mem_config = this->output_mem_config;
        if (!mem_config.shard_spec.has_value()) {
            ShardSpec shard_spec = input_tensor.shard_spec().value();
            if (mem_config.memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
                shard_spec.shape[1] = shard_spec.shape[1] * this->ring_size;
            } else {
                shard_spec.shape[0] = shard_spec.shape[0] * this->ring_size;
            }
        }
        const auto output_shape = this->compute_output_shapes(input_tensors)[0];
        return {create_sharded_device_tensor(
                output_shape,
                input_tensor.dtype(),
                input_tensor.layout(),
                input_tensor.device(),
                mem_config)};
    } else {
        return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), input_tensor.layout(), this->output_mem_config);
    }
}

operation::ProgramWithCallbacks AllGather::create_program(const std::vector<Tensor> & input_tensors, std::vector<Tensor> &output_tensors) const {
    if (input_tensors[0].is_sharded()) {
        return all_gather_multi_core_sharded(input_tensors[0], output_tensors[0], this->dim, this->num_links, this->ring_size, this->ring_index, this->receiver_device_id, this->sender_device_id);
    } else {
        return all_gather_multi_core_with_workers(input_tensors[0], output_tensors[0], this->dim, this->num_links, this->ring_size, this->ring_index, this->receiver_device_id, this->sender_device_id);
    }
}

tt::stl::reflection::Attributes AllGather::attributes() const {
    return {
        {"dim", this->dim},
        {"num_links", this->num_links},
        {"ring_size", this->ring_size},
        {"ring_index", this->ring_index},
        {"receiver_device_id", this->receiver_device_id},
        {"sender_device_id", this->sender_device_id},
        {"output_mem_config", this->output_mem_config},
    };
}

std::vector<Tensor> all_gather(const std::vector<Tensor>& input_tensors, const uint32_t dim, const uint32_t num_links, const MemoryConfig& output_mem_config) {

    TT_FATAL(std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr, "This op is only supported for Fast Dispatch");

    std::vector<Tensor> output_tensors;
    output_tensors.reserve(input_tensors.size());
    // Temporary changes to allow multi-device ops to work with op profiler
    // Should be removed with new profiler + software queue changes
    tt:tt_metal::operation::skip_profile = getDeviceProfilerState();
    std::vector<AllGather> ops;
    ops.reserve(input_tensors.size());
    for (uint32_t i = 0; i < input_tensors.size(); ++i) {
        chip_id_t receiver_device_id = input_tensors[(i + 1) % input_tensors.size()].device()->id();
        chip_id_t sender_device_id = input_tensors[i == 0 ? input_tensors.size() - 1 : i - 1].device()->id();
        ops.emplace_back(AllGather{dim, num_links, static_cast<uint32_t>(input_tensors.size()), i, receiver_device_id, sender_device_id, output_mem_config});
        output_tensors.push_back(operation::run(ops[i], {input_tensors[i]}).at(0));
    }
    if (tt::tt_metal::operation::skip_profile) {
        for (uint32_t i = 0; i < input_tensors.size(); ++i) {
            const auto& operation = ops[i];
            const std::vector<Tensor> inputs = {input_tensors[i]};
            const std::vector<Tensor> outputs = {output_tensors[i]};
            const auto& program = operation::skipped_programs.at(input_tensors[i].device()->id());

            tt::tt_metal::operation::ProfilerInfo profiler_info = {.preferred_name = "tt::tt_metal::AllGather", .parallelization_strategy = std::nullopt};
            auto profile_scope = op_profiler::OpProfileScope(profiler_info.preferred_name.value(), op_profiler::OpType::tt_dnn_device);
            auto do_profile = op_profiler::get_profiler_flag();
            if (do_profile) {
                if (profiler_info.preferred_name.has_value()) {
                    op_profiler::set_preferred_name(profiler_info.preferred_name.value());
                }
                if (profiler_info.parallelization_strategy.has_value()) {
                    op_profiler::set_parallelization_strategy(profiler_info.parallelization_strategy.value());
                }
                op_profiler::append_math_fidelities(program);
                op_profiler::append_meta_data(fmt::format("{}", operation.attributes()));
            }
            op_profiler::dump_device_profiler_results(input_tensors[i].device(), program);
            op_profiler::append_all_tensor_io_data(inputs, {}, outputs);
        }
        tt::tt_metal::operation::skip_profile = false;
        operation::skipped_programs.clear();
    }
    return output_tensors;
}

}  // namespace tt_metal

}  // namespace tt
