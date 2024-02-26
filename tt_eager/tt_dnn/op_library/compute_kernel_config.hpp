// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt {
namespace tt_metal {

struct GrayskullComputeKernelConfig {
    MathFidelity math_fidelity = MathFidelity::LoFi;
    bool math_approx_mode = true;
};

struct WormholeComputeKernelConfig {
    MathFidelity math_fidelity = MathFidelity::LoFi;
    bool math_approx_mode = true;
    bool fp32_dest_acc_en = false;
    bool packer_l1_acc = false;
};

using DeviceComputeKernelConfig = std::variant<GrayskullComputeKernelConfig, WormholeComputeKernelConfig>;

inline DeviceComputeKernelConfig init_device_compute_kernel_config(
    ARCH arch,
    std::optional<const DeviceComputeKernelConfig>& device_kernel_config,
    MathFidelity default_fidelity=MathFidelity::LoFi,
    bool default_approx_mode=true,
    bool default_fp32_acc=false,
    bool default_l1_acc=false)
{
    DeviceComputeKernelConfig defaultConfig;

    if (device_kernel_config.has_value()) {
        auto compute_kernel_config = device_kernel_config.value();
        std::visit([&](auto&& compute_kernel_config) {
            using T = std::decay_t<decltype(compute_kernel_config)>;
            if constexpr (std::is_same_v<T, GrayskullComputeKernelConfig>) {
                TT_ASSERT(arch == ARCH::GRAYSKULL, "kernel config is not for graykull");
                MathFidelity math_fidelity = compute_kernel_config.math_fidelity;
                bool math_approx_mode = compute_kernel_config.math_approx_mode;
                defaultConfig = GrayskullComputeKernelConfig{.math_fidelity = math_fidelity, .math_approx_mode = math_approx_mode};
            } else if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
                TT_ASSERT(arch == ARCH::WORMHOLE_B0, "kernel config is not for wormhole_b0");
                MathFidelity math_fidelity = compute_kernel_config.math_fidelity;
                bool math_approx_mode = compute_kernel_config.math_approx_mode;
                bool fp32_dest_acc_en = compute_kernel_config.fp32_dest_acc_en;
                bool packer_l1_acc = compute_kernel_config.packer_l1_acc;
                defaultConfig = WormholeComputeKernelConfig{.math_fidelity = math_fidelity, .math_approx_mode = math_approx_mode, .fp32_dest_acc_en = fp32_dest_acc_en, .packer_l1_acc = packer_l1_acc};
            } else {
                TT_FATAL("arch not supported");
            }
        }, compute_kernel_config);
        return defaultConfig;
    } else {
        if (arch == ARCH::GRAYSKULL) {
            return GrayskullComputeKernelConfig{.math_fidelity = default_fidelity, .math_approx_mode = default_approx_mode};
        } else {
            return WormholeComputeKernelConfig{.math_fidelity = default_fidelity, .math_approx_mode = default_approx_mode, .fp32_dest_acc_en = default_fp32_acc, .packer_l1_acc = default_l1_acc};
        }
    }
}

}
}
