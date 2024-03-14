// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tensor/tensor_impl.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

struct MorehGroupNormBackwardInputGrad {
    uint32_t num_groups;
    MemoryConfig input_grad_mem_config;

    void validate_with_output_tensors(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        const std::vector<std::optional<Tensor>> &output_tensors) const;

    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;

    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;

    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple("num_groups", "input_grad_mem_config");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->num_groups), std::cref(this->input_grad_mem_config));
    }
};

operation::ProgramWithCallbacks moreh_groupnorm_backward_input_grad_impl(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mean,
    const Tensor &rstd,
    uint32_t num_groups,
    Tensor &input_grad,
    const std::optional<const Tensor> gamma = std::nullopt);

Tensor moreh_groupnorm_backward_input_grad(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mean,
    const Tensor &rstd,
    uint32_t num_groups,
    const std::optional<const Tensor> gamma = std::nullopt,
    const std::optional<const Tensor> input_grad = std::nullopt,
    const MemoryConfig &input_grad_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

struct MorehGroupNormBackwardGammaBetaGrad {
    uint32_t num_groups;
    MemoryConfig gamma_grad_mem_config;
    MemoryConfig beta_grad_mem_config;
    bool gamma_requires_grad;
    bool beta_requires_grad;

    void validate_with_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;

    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;

    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;

    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple(
        "num_groups", "gamma_grad_mem_config", "beta_grad_mem_config", "gamma_requires_grad", "beta_requires_grad");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->num_groups),
            std::cref(this->gamma_grad_mem_config),
            std::cref(this->beta_grad_mem_config),
            std::cref(this->gamma_requires_grad),
            std::cref(this->beta_requires_grad));
    }
};

operation::ProgramWithCallbacks moreh_groupnorm_backward_gamma_beta_grad_impl(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mean,
    const Tensor &rstd,
    uint32_t num_groups,
    const std::optional<Tensor> dgamma = std::nullopt,
    const std::optional<Tensor> dbeta = std::nullopt);

std::vector<std::variant<Tensor, char *>> moreh_groupnorm_backward_gamma_beta_grad(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mean,
    const Tensor &rstd,
    uint32_t num_groups,
    const std::optional<const Tensor> gamma_grad = std::nullopt,
    const std::optional<const Tensor> beta_grad = std::nullopt,
    const MemoryConfig &gamma_grad_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const MemoryConfig &beta_grad_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    bool gamma_requires_grad = true,
    bool beta_requires_grad = true);

std::vector<std::variant<Tensor, char *>> moreh_groupnorm_backward(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mean,
    const Tensor &rstd,
    uint32_t num_groups,
    const std::optional<const Tensor> gamma = std::nullopt,
    const std::optional<const Tensor> input_grad = std::nullopt,
    const std::optional<const Tensor> gamma_grad = std::nullopt,
    const std::optional<const Tensor> beta_grad = std::nullopt,
    const MemoryConfig &input_grad_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const MemoryConfig &gamma_grad_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const MemoryConfig &beta_grad_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    bool input_requires_grad = true,
    bool gamma_requires_grad = true,
    bool beta_requires_grad = true);

}  // namespace primary

}  // namespace operations

}  // namespace tt
