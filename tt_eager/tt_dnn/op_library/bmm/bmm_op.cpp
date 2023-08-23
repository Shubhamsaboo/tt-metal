// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

#include <optional>
#include <algorithm>

using namespace tt::constants;

vector<uint32_t> _get_prime_factors(uint32_t n) {
    uint32_t i = 2;

    vector<uint32_t> prime_factors;
    while (i * i <= n) {
        if (n % i != 0) i++;
        else {
            n /= i;
            prime_factors.push_back(i);
        }
    }
    if (n > 1) prime_factors.push_back(n);

    return prime_factors;
}

vector<uint32_t> _get_possible_products(vector<uint32_t> factors) {
    if (factors.size() == 0) return {1};

    vector<uint32_t> products;
    for (uint32_t& fac : factors) {
        vector<uint32_t> new_products;
        if (not std::count(products.begin(), products.end(), fac))
            new_products.push_back(fac);
        for (uint32_t& prod : products) {
            if (not std::count(products.begin(), products.end(), fac * prod))
                new_products.push_back(fac * prod);
        }

        // Insert all new products to product
        products.reserve(products.size() + distance(new_products.begin(), new_products.end()));
        products.insert(products.end(), new_products.begin(), new_products.end());
    }

    // Sort products
    std::sort(products.begin(), products.end());

    return products;
}

uint32_t _get_maximum_block_dim(int32_t block_dim, int32_t in0_block_w) {
    int32_t other_dim = (400 - 2 * in0_block_w * block_dim) / (2 * in0_block_w + block_dim);
    if (other_dim > 0)
        return other_dim;
    return 0;
}

namespace bmm_op_utils {
using namespace tt;
using namespace tt::tt_metal;


tuple<uint32_t, uint32_t, uint32_t, uint32_t> get_large_matmul_params(uint32_t Mt, uint32_t Nt, uint32_t num_cores_y, uint32_t num_cores_x, uint32_t in0_block_w) {
    auto Nt_fac = _get_prime_factors(Nt);
    auto Mt_fac = _get_prime_factors(Mt);
    uint32_t Npc_min = 1;
    uint32_t Mpc_min = 1;

    for (auto it = Nt_fac.begin(); it != Nt_fac.end(); ++it) {
        auto ele = *it;
        if (ele > num_cores_x) {
            Npc_min *= ele;
            Nt_fac.erase(it);
            --it;
        }
    }
    for (auto it = Mt_fac.begin(); it != Mt_fac.end(); ++it) {
        auto ele = *it;
        if (ele > num_cores_y) {
            Mpc_min *= ele;
            Mt_fac.erase(it);
            --it;
        }
    }

    if (Npc_min > _get_maximum_block_dim(Mpc_min, in0_block_w))
        return {0, 0, 0, 0};

    uint32_t Mpc = Mpc_min;
    uint32_t Npc = Npc_min;
    if (Mpc_min > 1) {
        auto Npc_choices = _get_possible_products(Nt_fac);
        auto Npc_max = _get_maximum_block_dim(Mpc_min, in0_block_w);
        for (auto &ele : Npc_choices) {
            if (ele *  Npc_min <= Npc_max)
                Npc = ele * Npc_min;
            else
                break;
        }

        if (Mt / Mpc > num_cores_y or Nt / Npc > num_cores_x)
            return {0, 0, 0, 0};

        for (auto &subblock_hw : SUBBLOCK_HW_CHOICES) {
            auto subblock_h = std::get<0>(subblock_hw);
            auto subblock_w = std::get<1>(subblock_hw);
            if (Mpc % subblock_h == 0 and Npc % subblock_w == 0)
                return {Mpc, Npc, subblock_h, subblock_w};
        }
    }

    else if (Npc_min > 1) {
        auto Mpc_choices = _get_possible_products(Mt_fac);
        auto Mpc_max = _get_maximum_block_dim(Npc_min, in0_block_w);
        for (auto &ele : Mpc_choices) {
            if (ele *  Mpc_min <= Mpc_max)
                Mpc = ele * Mpc_min;
            else
                break;
        }

        if (Mt / Mpc > num_cores_y or Nt / Npc > num_cores_x) {
            return {0, 0, 0, 0};
        }

        for (auto &subblock_hw : SUBBLOCK_HW_CHOICES) {
            auto subblock_h = std::get<0>(subblock_hw);
            auto subblock_w = std::get<1>(subblock_hw);
            if (Mpc % subblock_h == 0 and Npc % subblock_w == 0)
                return {Mpc, Npc, subblock_h, subblock_w};
        }
    }

    else {
        auto Mpc_choices = _get_possible_products(Mt_fac);
        auto Npc_choices = _get_possible_products(Nt_fac);
        for (auto &Npc : Npc_choices) {
            auto Mpc_max = _get_maximum_block_dim(Npc, in0_block_w);
            for (auto &ele : Mpc_choices) {
                if (ele <= Mpc_max)
                    Mpc = ele;
            }

            if (Mt / Mpc > num_cores_y or Nt / Npc > num_cores_x)
                continue;

            for (auto &subblock_hw : SUBBLOCK_HW_CHOICES) {
                auto subblock_h = std::get<0>(subblock_hw);
                auto subblock_w = std::get<1>(subblock_hw);
                if (Mpc % subblock_h == 0 and Npc % subblock_w == 0)
                    return {Mpc, Npc, subblock_h, subblock_w};
            }
        }
    }

    return {0, 0, 0, 0};
}


CoreCoord get_core_range(uint32_t num_blocks_rows, uint32_t num_blocks_cols, uint32_t max_num_rows, uint32_t max_num_cols) {
    CoreCoord core_range(0, 0);
    if (!(num_blocks_rows == 1 && num_blocks_cols == 1) && num_blocks_rows <= max_num_rows && num_blocks_cols <= max_num_cols) {
        core_range.x = num_blocks_cols;
        core_range.y = num_blocks_rows;
    }
    return core_range;
}

MatmulParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto& ashape = input_tensor_a.shape(), bshape = input_tensor_b.shape();
    uint32_t num_output_tiles = ashape[0] * ashape[1] * ashape[2] * bshape[3] / TILE_HW; // Output M x N

    // Parameters for large matmul with reuse
    uint32_t B = ashape[0] * ashape[1];
    uint32_t Mt = ashape[2]/TILE_HEIGHT;
    uint32_t Kt = ashape[3]/TILE_WIDTH;
    uint32_t Nt = bshape[3]/TILE_WIDTH;
    uint32_t in0_block_w = 2;

    tt::tt_metal::Device *device = input_tensor_a.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    bool use_general_large_matmul_params = false; // Hard force to use default 16, 16, 4, 2
    uint32_t per_core_M, per_core_N, out_subblock_h, out_subblock_w;
    uint32_t num_blocks_x, num_blocks_y;
    if (use_general_large_matmul_params) {
        // Get large matmul params
        auto matmul_params = bmm_op_utils::get_large_matmul_params(Mt, Nt, num_cores_y, num_cores_x, in0_block_w);
        per_core_M = std::get<0>(matmul_params);
        per_core_N = std::get<1>(matmul_params);
        out_subblock_h = std::get<2>(matmul_params);
        out_subblock_w = std::get<3>(matmul_params);
    }
    else {
        // out_subblock h/w doesn't matter
        per_core_M = 16;
        per_core_N = 16;

        // Calculate number of blocks along x and y; tensor dims are padded up to 512
        num_blocks_y = (Mt - 1) / per_core_M + 1;
        num_blocks_x = (Nt - 1) / per_core_N + 1;
    }

    // If no possible params, matmul_params will be (0, 0, 0, 0)
    if (use_general_large_matmul_params and per_core_M > 0 and Kt % in0_block_w == 0 and B == 1) {
        CoreCoord core_range = get_core_range((Mt / per_core_M), (Nt / per_core_N), num_cores_y, num_cores_x);
        // If matmul params are (16, 16, 4, 2), use the default mcast op
        if (
            per_core_M == 16 and
            per_core_N == 16 and
            out_subblock_h == 4 and
            out_subblock_w == 2
        ) {
            if (core_range.y > 0)
                return MatmulParallelizationStrategy::MULTI_CORE_REUSE_MCAST;
            return MatmulParallelizationStrategy::MULTI_CORE_REUSE;
        }
        else if (core_range.y > 0)
            return MatmulParallelizationStrategy::MULTI_CORE_REUSE_MCAST_GENERALIZED;
        return MatmulParallelizationStrategy::MULTI_CORE_REUSE_GENERALIZED;
    }
    else if (num_blocks_x * num_blocks_y <= num_cores_x * num_cores_y and Kt % in0_block_w == 0) {
        CoreCoord core_range = get_core_range(num_blocks_y, num_blocks_x, num_cores_y, num_cores_x);
        // If we don't need padding, use the default multi_core reuse/reuse_mcast
        if (Mt % per_core_M == 0 and Nt % per_core_N == 0) {
            if (core_range.y > 0)
                return MatmulParallelizationStrategy::MULTI_CORE_REUSE_MCAST;
            return MatmulParallelizationStrategy::MULTI_CORE_REUSE;
        }
        else if (core_range.y > 0)
            return MatmulParallelizationStrategy::MULTI_CORE_REUSE_MCAST_PADDING;
        return MatmulParallelizationStrategy::MULTI_CORE;
    }
    else if (num_output_tiles > 1) {
        return MatmulParallelizationStrategy::MULTI_CORE;
    }
    else {
        return MatmulParallelizationStrategy::SINGLE_CORE;
    }
}

tt::operations::primary::MatmulMultiCoreReuseMultiCast1DProgramConfig get_mcast_1d_config(const Tensor &input_tensor_a, const Tensor &input_tensor_b, bool fuse_batch, bool fuse_gelu_activation) {
    auto device = input_tensor_a.device();
    auto grid_size = device->compute_with_storage_grid_size();
    uint32_t M = fuse_batch ? input_tensor_a.shape()[0] * input_tensor_a.shape()[1] * input_tensor_a.shape()[-2] : input_tensor_a.shape()[-2];
    uint32_t per_core_M = M / TILE_HEIGHT;
    uint32_t per_core_N = div_up(div_up(input_tensor_b.shape()[-1], grid_size.x * grid_size.y), TILE_WIDTH);
    uint32_t out_subblock_h, out_subblock_w;
    bool params_found = false;
    for (auto &subblock_hw : bmm_op_utils::SUBBLOCK_HW_CHOICES) {
        out_subblock_h = std::get<0>(subblock_hw);
        out_subblock_w = std::get<1>(subblock_hw);
        if (per_core_M % out_subblock_h == 0 and per_core_N % out_subblock_w == 0) {
            params_found = true;
            break;
        }
    }
    TT_ASSERT(params_found, "Matmul parameters could not be determined for given input shapes");

    return tt::operations::primary::MatmulMultiCoreReuseMultiCast1DProgramConfig{
        .compute_with_storage_grid_size = grid_size,
        .in0_block_w = 2,
        .out_subblock_h = out_subblock_h,
        .out_subblock_w = out_subblock_w,
        .per_core_M = per_core_M,
        .per_core_N = per_core_N,
        .fuse_batch = fuse_batch,
        .fuse_gelu_activation = fuse_gelu_activation,
    };
}

}

namespace tt {
namespace tt_metal {

void Matmul::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    TT_ASSERT((input_tensor_a.layout() == Layout::TILE && input_tensor_b.layout() == Layout::TILE), "Inputs to matmul must be tilized");
    if (this->bcast_batch) {
        TT_ASSERT(input_tensor_b.shape()[0] * input_tensor_b.shape()[1] == 1 && "matmul (batch bcast variant) expects input tensors of shapes BCMK*11KN=BCMN");
    } else {
        TT_ASSERT(input_tensor_a.shape()[1] == input_tensor_b.shape()[1] && input_tensor_a.shape()[0] == input_tensor_b.shape()[0] && "bmm (non-bcast matmul) expects input tensors of shapes BCMK*BCKN=BCMN");
    }
    TT_ASSERT(input_tensor_a.shape()[3] == input_tensor_b.shape()[2] && "Dimension K (A.shape[3] and B.shape[2]) must match for A and B in bmm_op"); // A.K == B.K

    // TODO: Uplift get_parallelization_strategy to be struct param? We should do separate dtype validations for different parallelizations
    // This requires sweeping across shapes with different dtypes/dataformats; for now, ignore dtype assertions here and uplift to actual matmul/bmm implementations
    TT_ASSERT(input_tensor_a.dtype() == tt::tt_metal::DataType::BFLOAT16 || input_tensor_a.dtype() == tt::tt_metal::DataType::BFLOAT8_B, "Unsupported data format");
    TT_ASSERT(input_tensor_a.storage_type() == StorageType::DEVICE and input_tensor_b.storage_type() == StorageType::DEVICE, "Operands to matmul need to be on device!");
    TT_ASSERT(input_tensor_a.device() == input_tensor_b.device(), "Operands to matmul need to be on the same device!");
    TT_ASSERT(input_tensor_a.buffer() != nullptr and input_tensor_b.buffer() != nullptr, "Operands to matmul need to be allocated in buffers on device!");
}

std::vector<Shape> Matmul::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    auto output_shape = input_tensor_a.shape();
    output_shape[-1] = input_tensor_b.shape()[-1];
    return {output_shape};
}

std::vector<Tensor> Matmul::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, this->output_dtype, Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks Matmul::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    auto& output_tensor = output_tensors.at(0);

    auto parallelization_strategy = this->get_parallelization_strategy(input_tensors);

    switch (parallelization_strategy){
        case MatmulParallelizationStrategy::MULTI_CORE:
            return matmul_multi_core(input_tensor_a, input_tensor_b, output_tensor, this->bcast_batch);
        case MatmulParallelizationStrategy::MULTI_CORE_REUSE:
            return matmul_multi_core_reuse(input_tensor_a, input_tensor_b, output_tensor, this->bcast_batch);
        case MatmulParallelizationStrategy::MULTI_CORE_REUSE_MCAST:
            return matmul_multi_core_reuse_mcast(input_tensor_a, input_tensor_b, output_tensor, this->bcast_batch);
        case MatmulParallelizationStrategy::MULTI_CORE_REUSE_GENERALIZED:
            return matmul_multi_core_reuse_generalized(input_tensor_a, input_tensor_b, output_tensor, this->bcast_batch);
        case MatmulParallelizationStrategy::MULTI_CORE_REUSE_MCAST_GENERALIZED:
            return matmul_multi_core_reuse_mcast_generalized(input_tensor_a, input_tensor_b, output_tensor, this->bcast_batch);
        case MatmulParallelizationStrategy::MULTI_CORE_REUSE_PADDING:
            return matmul_multi_core_reuse_padding(input_tensor_a, input_tensor_b, output_tensor, this->bcast_batch);
        case MatmulParallelizationStrategy::MULTI_CORE_REUSE_MCAST_PADDING:
            return matmul_multi_core_reuse_mcast_padding(input_tensor_a, input_tensor_b, output_tensor, this->bcast_batch);
        case MatmulParallelizationStrategy::SINGLE_CORE:
        default:
            return matmul_single_core(input_tensor_a, input_tensor_b, output_tensor, this->bcast_batch);
    }

}

tt::stl::reflection::Attributes Matmul::attributes() const {
    return {
        {"bcast_batch", this->bcast_batch},
        {"output_mem_config", this->output_mem_config},
        {"output_dtype", this->output_dtype},
    };
}

MatmulParallelizationStrategy Matmul::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    return bmm_op_utils::get_parallelization_strategy(input_tensors);
}

/**
 * Bert large matmuls using operations::primary::matmul + program_config
 */
Tensor bert_large_fused_qkv_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype) {
    auto batch_size = input_tensor_a.shape()[0];

    TT_ASSERT((input_tensor_a.shape() == Shape({batch_size, 1, 384, 1024})), "Unsupported input shape");
    TT_ASSERT((input_tensor_b.shape() == Shape({1, 1, 1024, 3072})), "Unsupported input shape");

    auto program_config = operations::primary::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = {12, batch_size},
        .in0_block_w = 4,
        .out_subblock_h = 4,
        .out_subblock_w = 2,
        .per_core_M = 12,
        .per_core_N = 8,
        .fused_activation = std::nullopt,
    };
    return operations::primary::matmul(input_tensor_a, input_tensor_b, bias, program_config, mem_config, output_dtype);
}

Tensor bert_large_ff1_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, std::optional<UnaryWithParam> fused_activation, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype) {
    auto batch_size = input_tensor_a.shape()[0];

    TT_ASSERT((input_tensor_a.dtype() != DataType::BFLOAT16 or input_tensor_b.dtype() != DataType::BFLOAT16 or output_dtype != DataType::BFLOAT16) or (mem_config.buffer_type == BufferType::DRAM) or (input_tensor_a.memory_config().buffer_type == BufferType::DRAM and input_tensor_b.memory_config().buffer_type == BufferType::DRAM), "For BFLOAT16, if output is on L1, one of in0 or in1 must be on DRAM!");
    TT_ASSERT((input_tensor_a.shape() == Shape({batch_size, 1, 384, 1024})), "Unsupported input shape");
    TT_ASSERT((input_tensor_b.shape() == Shape({1, 1, 1024, 4096})), "Unsupported input shape");

    auto program_config = operations::primary::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = {12, batch_size},
        .in0_block_w = 4,
        .out_subblock_h = 6,
        .out_subblock_w = 1,
        .per_core_M = 12,
        .per_core_N = 11,
        .fused_activation = fused_activation,
    };
    return operations::primary::matmul(input_tensor_a, input_tensor_b, bias, program_config, mem_config, output_dtype);
}

Tensor bert_large_ff2_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype) {
    auto batch_size = input_tensor_a.shape()[0];

    TT_ASSERT((input_tensor_a.shape() == Shape({batch_size, 1, 384, 4096})), "Unsupported input shape");
    TT_ASSERT((input_tensor_b.shape() == Shape({1, 1, 4096, 1024})), "Unsupported input shape");

    auto program_config = operations::primary::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = {12, batch_size},
        .in0_block_w = 4,
        .out_subblock_h = 6,
        .out_subblock_w = 1,
        .per_core_M = 12,
        .per_core_N = 3,
        .fused_activation = std::nullopt,
    };
    return operations::primary::matmul(input_tensor_a, input_tensor_b, bias, program_config, mem_config, output_dtype);
}

Tensor bert_large_selfout_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype) {
    auto batch_size = input_tensor_a.shape()[0];

    TT_ASSERT((input_tensor_a.shape() == Shape({batch_size, 1, 384, 1024})), "Unsupported input shape");
    TT_ASSERT((input_tensor_b.shape() == Shape({1, 1, 1024, 1024})), "Unsupported input shape");

    auto program_config = operations::primary::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = {12, batch_size},
        .in0_block_w = 4,
        .out_subblock_h = 6,
        .out_subblock_w = 1,
        .per_core_M = 12,
        .per_core_N = 3,
        .fused_activation = std::nullopt,
    };
    return operations::primary::matmul(input_tensor_a, input_tensor_b, bias, program_config, mem_config, output_dtype);
}

Tensor bert_large_pre_softmax_bmm(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype) {
    auto batch_size = input_tensor_a.shape()[0];

    TT_ASSERT((input_tensor_a.shape() == Shape({batch_size, 16, 384, 64})), "Unsupported input shape");
    TT_ASSERT((input_tensor_b.shape() == Shape({batch_size, 16, 64, 384})), "Unsupported input shape");

    auto program_config = operations::primary::MatmulMultiCoreReuseProgramConfig{
        .compute_with_storage_grid_size = {12, batch_size},
        .in0_block_w = 1,
        .out_subblock_h = 4,
        .out_subblock_w = 2,
        .per_core_M = 12,
        .per_core_N = 12,
    };
    return operations::primary::matmul(input_tensor_a, input_tensor_b, program_config, mem_config, output_dtype);

}

Tensor bert_large_post_softmax_bmm(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype) {
    auto batch_size = input_tensor_a.shape()[0];

    TT_ASSERT((input_tensor_a.shape() == Shape({batch_size, 16, 384, 384})), "Unsupported input shape");
    TT_ASSERT((input_tensor_b.shape() == Shape({batch_size, 16, 384, 64})), "Unsupported input shape");

    auto program_config = operations::primary::MatmulMultiCoreReuseProgramConfig{
        .compute_with_storage_grid_size = {12, batch_size},
        .in0_block_w = 2,
        .out_subblock_h = 4,
        .out_subblock_w = 2,
        .per_core_M = 12,
        .per_core_N = 2,
    };
    return operations::primary::matmul(input_tensor_a, input_tensor_b, program_config, mem_config, output_dtype);

}

/**
 * Falcon matmuls using operations::primary::matmul + program_config
 */
Tensor falcon_fused_qkv_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype) {
    auto program_config = bmm_op_utils::get_mcast_1d_config(input_tensor_a, input_tensor_b, true);
    return operations::primary::matmul_1d(input_tensor_a, input_tensor_b, bias, program_config, mem_config, output_dtype);
}

Tensor falcon_selfout_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype) {
    auto program_config = bmm_op_utils::get_mcast_1d_config(input_tensor_a, input_tensor_b, true);
    return operations::primary::matmul_1d(input_tensor_a, input_tensor_b, bias, program_config, mem_config, output_dtype);
}

Tensor falcon_dense_4h_to_h_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype) {
    auto program_config = bmm_op_utils::get_mcast_1d_config(input_tensor_a, input_tensor_b, true);
    return operations::primary::matmul_1d(input_tensor_a, input_tensor_b, bias, program_config, mem_config, output_dtype);
}

Tensor falcon_dense_h_to_4h_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, bool fuse_gelu_activation, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype) {
    auto seq_len = input_tensor_a.shape()[2];
    if (seq_len > 1024) {
        // TODO: Check support for seq_len == 128, 256, 512, ..., 2048
        TT_ASSERT(seq_len % TILE_HEIGHT == 0, "Falcon mm's seq_len must be a multiple of 32!");
        TT_ASSERT(seq_len >=  128, "Falcon mm's seq_len must be greater than 128!");
        TT_ASSERT((input_tensor_a.shape() == Shape({1, 1, seq_len, 4544})), "Unsupported input shape");
        TT_ASSERT((input_tensor_b.shape() == Shape({1, 1, 4544, 18176})), "Unsupported input shape");
        return operation::run_with_autoformat(Matmul{.bcast_batch=true, .output_mem_config=mem_config, .output_dtype=output_dtype.value_or(input_tensor_a.dtype())}, {input_tensor_a, input_tensor_b}).at(0);
    } else {
        auto program_config = bmm_op_utils::get_mcast_1d_config(input_tensor_a, input_tensor_b, true, fuse_gelu_activation);
        return operations::primary::matmul_1d(input_tensor_a, input_tensor_b, bias, program_config, mem_config, output_dtype);
    }
}

Tensor falcon_lm_head_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype) {
    auto seq_len = input_tensor_a.shape()[2];

    if (seq_len > 512) {
        // TODO: Check support for seq_len == 128, 256, 512, ..., 2048
        TT_ASSERT(seq_len % TILE_HEIGHT == 0, "Falcon mm's seq_len must be a multiple of 32!");
        TT_ASSERT(seq_len >=  128, "Falcon mm's seq_len must be greater than 128!");
        TT_ASSERT((input_tensor_a.shape() == Shape({1, 1, seq_len, 4544})), "Unsupported input shape");
        TT_ASSERT((input_tensor_b.shape() == Shape({1, 1, 4544, 65024})), "Unsupported input shape");
        return operation::run_with_autoformat(Matmul{.bcast_batch=true, .output_mem_config=mem_config, .output_dtype=output_dtype.value_or(input_tensor_a.dtype())}, {input_tensor_a, input_tensor_b}).at(0);
    } else {
        auto program_config = bmm_op_utils::get_mcast_1d_config(input_tensor_a, input_tensor_b, true);
        return operations::primary::matmul_1d(input_tensor_a, input_tensor_b, bias, program_config, mem_config, output_dtype);
    }
}

}  // namespace tt_metal



namespace operations {

namespace primary {


tt::stl::reflection::Attributes MatmulMultiCoreReuseProgramConfig::attributes() const {
    return {
        {"compute_with_storage_grid_size",  this->compute_with_storage_grid_size.str()},
        {"in0_block_w",  this->in0_block_w},
        {"out_subblock_h",  this->out_subblock_h},
        {"out_subblock_w",  this->out_subblock_w},
        {"per_core_M",  this->per_core_M},
        {"per_core_N",  this->per_core_N},
    };
}

tt::stl::reflection::Attributes MatmulMultiCoreReuseMultiCastProgramConfig::attributes() const {
    return {
        {"compute_with_storage_grid_size",  this->compute_with_storage_grid_size.str()},
        {"in0_block_w",  this->in0_block_w},
        {"out_subblock_h",  this->out_subblock_h},
        {"out_subblock_w",  this->out_subblock_w},
        {"per_core_M",  this->per_core_M},
        {"per_core_N",  this->per_core_N},
        {"fusee_activation",  this->fused_activation},
    };
}

tt::stl::reflection::Attributes MatmulMultiCoreReuseMultiCast1DProgramConfig::attributes() const {
    return {
        {"compute_with_storage_grid_size",  this->compute_with_storage_grid_size.str()},
        {"in0_block_w",  this->in0_block_w},
        {"out_subblock_h",  this->out_subblock_h},
        {"out_subblock_w",  this->out_subblock_w},
        {"per_core_M",  this->per_core_M},
        {"per_core_N",  this->per_core_N},
        {"fuse_batch",  this->fuse_batch},
        {"fuse_gelu_activation",  this->fuse_gelu_activation},
    };
}

void Matmul::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors
) const {

    TT_ASSERT(input_tensors.size() == 2);
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    TT_ASSERT((input_tensor_a.layout() == Layout::TILE && input_tensor_b.layout() == Layout::TILE), "Inputs to matmul must be tilized");
    TT_ASSERT(input_tensor_a.shape()[3] == input_tensor_b.shape()[2] && "Dimension K (A.shape[3] and B.shape[2]) must match for A and B in bmm_op"); // A.K == B.K

    TT_ASSERT(input_tensor_a.dtype() == tt::tt_metal::DataType::BFLOAT16 || input_tensor_a.dtype() == tt::tt_metal::DataType::BFLOAT8_B, "Unsupported data format");
    TT_ASSERT(input_tensor_a.storage_type() == StorageType::DEVICE and input_tensor_b.storage_type() == StorageType::DEVICE, "Operands to matmul need to be on device!");
    TT_ASSERT(input_tensor_a.device() == input_tensor_b.device(), "Operands to matmul need to be on the same device!");
    TT_ASSERT(input_tensor_a.buffer() != nullptr and input_tensor_b.buffer() != nullptr, "Operands to matmul need to be allocated in buffers on device!");


    TT_ASSERT(optional_input_tensors.size() == 1);
    const auto& optional_bias = optional_input_tensors.at(0);
    if (optional_bias.has_value()) {
        const auto& bias = optional_bias.value();
        TT_ASSERT(bias.layout() == Layout::TILE, "Unsupported input layout");
        TT_ASSERT(bias.shape() == Shape({1, 1, TILE_HEIGHT, input_tensor_b.shape()[3]}), "Unsupported bias shape");
    }

    std::visit(
        [&](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (
                std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseProgramConfig> ||
                std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCastProgramConfig> ||
                std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCast1DProgramConfig>
            ) {
                TT_ASSERT((input_tensor_a.shape()[-1] / TILE_WIDTH) % program_config.in0_block_w == 0, "Kt must be divisible by in0_block_w");
                TT_ASSERT(program_config.per_core_M % program_config.out_subblock_h == 0, "per_core_M must be divisible by out_subblock_h");
                TT_ASSERT(program_config.per_core_N % program_config.out_subblock_w == 0, "per_core_N must be divisible by out_subblock_w");
            }
            if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCastProgramConfig>) {
                if (program_config.fused_activation.has_value()) {
                    TT_ASSERT(program_config.fused_activation.value().op_type == UnaryOpType::GELU || program_config.fused_activation.value().op_type == UnaryOpType::RELU);
                }
            }
        },
        this->program_config
    );

}

std::vector<Shape> Matmul::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    Shape output_shape = input_tensor_a.shape();
    output_shape[-1] = input_tensor_b.shape()[-1];
    return {output_shape};
}

std::vector<Tensor> Matmul::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    return operation::generic_create_output_tensors(*this, input_tensors, this->output_dtype, Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks Matmul::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor> &output_tensors
) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto& bias = optional_input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    tt::tt_metal::DataType output_dtype = this->output_dtype;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fuse_batch = true;

    return std::visit(
        [&](const auto& program_config) -> operation::ProgramWithCallbacks {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<ProgramConfigType, MatmulDefaultProgramConfig>) {
                auto parallelization_strategy = bmm_op_utils::get_parallelization_strategy(input_tensors);

                auto broadcast_batch = input_tensor_b.shape()[0] * input_tensor_b.shape()[1] == 1;
                switch (parallelization_strategy){
                    case MatmulParallelizationStrategy::MULTI_CORE:
                        return matmul_multi_core(input_tensor_a, input_tensor_b, output_tensor, broadcast_batch);
                    case MatmulParallelizationStrategy::MULTI_CORE_REUSE:
                        return matmul_multi_core_reuse(input_tensor_a, input_tensor_b, output_tensor, broadcast_batch);
                    case MatmulParallelizationStrategy::MULTI_CORE_REUSE_MCAST:
                        return matmul_multi_core_reuse_mcast(input_tensor_a, input_tensor_b, output_tensor, broadcast_batch);
                    case MatmulParallelizationStrategy::MULTI_CORE_REUSE_GENERALIZED:
                        return matmul_multi_core_reuse_generalized(input_tensor_a, input_tensor_b, output_tensor, broadcast_batch);
                    case MatmulParallelizationStrategy::MULTI_CORE_REUSE_MCAST_GENERALIZED:
                        return matmul_multi_core_reuse_mcast_generalized(input_tensor_a, input_tensor_b, output_tensor, broadcast_batch);
                    case MatmulParallelizationStrategy::MULTI_CORE_REUSE_PADDING:
                        return matmul_multi_core_reuse_padding(input_tensor_a, input_tensor_b, output_tensor, broadcast_batch);
                    case MatmulParallelizationStrategy::MULTI_CORE_REUSE_MCAST_PADDING:
                        return matmul_multi_core_reuse_mcast_padding(input_tensor_a, input_tensor_b, output_tensor, broadcast_batch);
                    case MatmulParallelizationStrategy::SINGLE_CORE:
                    default:
                        return matmul_single_core(input_tensor_a, input_tensor_b, output_tensor, broadcast_batch);
                }
            }
            else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseProgramConfig>) {
                return bmm_multi_core_reuse_optimized(
                    input_tensor_a, input_tensor_b, input_tensor_a.shape(), input_tensor_b.shape(), output_tensor,
                    program_config.compute_with_storage_grid_size,
                    output_dtype, math_fidelity,
                    program_config.in0_block_w, program_config.out_subblock_h, program_config.out_subblock_w,
                    program_config.per_core_M, program_config.per_core_N, fuse_batch
                );
            }
            else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCastProgramConfig>) {
                return matmul_multi_core_reuse_mcast_2d_optimized(
                    input_tensor_a, input_tensor_b, bias, output_tensor,
                    program_config.compute_with_storage_grid_size,
                    output_dtype, math_fidelity,
                    program_config.in0_block_w, program_config.out_subblock_h, program_config.out_subblock_w,
                    program_config.per_core_M, program_config.per_core_N, fuse_batch, program_config.fused_activation
                );
            }
            else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                return matmul_multi_core_reuse_mcast_1d_optimized(
                    input_tensor_a, input_tensor_b, bias, output_tensor,
                    program_config.compute_with_storage_grid_size,
                    output_dtype, math_fidelity,
                    program_config.in0_block_w, program_config.out_subblock_h, program_config.out_subblock_w,
                    program_config.per_core_M, program_config.per_core_N, program_config.fuse_batch, program_config.fuse_gelu_activation
                );
            } else {
                TT_THROW("Unrecognized Config");
            }
        },
        this->program_config
    );
}

tt::stl::reflection::Attributes Matmul::attributes() const {
    return {
        {"program_config", this->program_config},
        {"output_mem_config",  this->output_mem_config},
        {"output_dtype", this->output_dtype},
    };
}

Tensor matmul_1d(const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::optional<const Tensor> bias, std::optional<MatmulMultiCoreReuseMultiCast1DProgramConfig> program_config, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype) {
    if (!program_config.has_value()) {
        program_config = bmm_op_utils::get_mcast_1d_config(input_tensor_a, input_tensor_b);
    }
    return operations::primary::matmul(input_tensor_a, input_tensor_b, bias, program_config.value(), mem_config, output_dtype);
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
