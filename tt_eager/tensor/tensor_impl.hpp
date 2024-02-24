// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/borrowed_buffer_functions.hpp"
#include "tensor/owned_buffer_functions.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_utils.hpp"
#include "tensor/types.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_stl/concepts.hpp"

#include <chrono>
#include <optional>
#include <thread>
#include "tensor/tensor_impl_wrapper.hpp"
namespace tt {

namespace tt_metal {

namespace tensor_impl {

std::array<uint32_t, 2> get_sharded_page_shape(Layout layout, const Shape& shape, DataType dtype, uint32_t num_shards, std::array<uint32_t, 2> shard_shape);

// -----------------------------------------------------------------------------------------------------------------------------------------------
// ===============================================================================================================================================
//                                                              Low Level APIs
// ===============================================================================================================================================
// -----------------------------------------------------------------------------------------------------------------------------------------------

// ======================================================================================
//                        Data type converters, packers, and unpackers
// ======================================================================================
// TODO(arakhmati): Should cast_vec be a generator?

template <typename OutputDataType, template<typename> typename BufferType, typename InputDataType>
std::vector<OutputDataType> cast_vec(const BufferType<InputDataType>& data_to_convert) {
    std::vector<OutputDataType> converted_data;
    for (auto datum : data_to_convert) {
        if constexpr (std::is_same_v<OutputDataType, float> and std::is_same_v<InputDataType, bfloat16>) {
            converted_data.push_back(datum.to_float());
        }
        else if constexpr (std::is_same_v<OutputDataType, uint32_t> and std::is_same_v<InputDataType, bfloat16>) {
            converted_data.push_back((uint32_t)datum.to_uint16());
        }
        else {
            converted_data.push_back(static_cast<OutputDataType>(datum));
        }
    }
    return converted_data;
}

// TODO(arakhmati): Should pack_vec_into_uint32_vec be a generator?
template <typename DataType, template <typename> typename BufferType>
std::vector<uint32_t> pack_vec_into_uint32_vec(const BufferType<DataType>& data_to_pack) {
    if constexpr (std::is_same_v<DataType, uint32_t>) {
        return std::vector(std::begin(data_to_pack), std::end(data_to_pack));
    } else if constexpr (std::is_same_v<DataType, uint16_t>) {
        std::vector<uint32_t> output;
        for (auto index = 0; index < data_to_pack.size(); index += 2) {
            auto value = data_to_pack[index + 1] << 16 | data_to_pack[index];
            output.push_back(value);
        }
        return output;
    } else if constexpr (std::is_same_v<DataType, bfloat16>) {
        auto bfloat16_vec = std::vector(std::begin(data_to_pack), std::end(data_to_pack));
        return pack_bfloat16_vec_into_uint32_vec(bfloat16_vec);
    } else if constexpr (std::is_same_v<DataType, float>) {
        std::vector<uint32_t> uint32_data;
        union float_uint32_convert {
            uint32_t u;
            float f;
            float_uint32_convert() : u(0) {}
        };
        for (auto i = 0; i < data_to_pack.size(); i ++) {
            float_uint32_convert a;
            a.f = data_to_pack[i];
            uint32_data.push_back(a.u);
        }
        return uint32_data;
    } else {
        static_assert(tt::stl::concepts::always_false_v<DataType>, "Don't know how to unpack uint32 data generically!");
    }
}

template <typename DataType>
std::vector<DataType> unpack_uint32_vec(std::vector<uint32_t>& data_to_unpack) {
    if constexpr (std::is_same_v<DataType, uint32_t>) {
        return data_to_unpack;
    } else if constexpr (std::is_same_v<DataType, uint16_t>) {
        std::vector<DataType> output;
        for (auto index = 0; index < data_to_unpack.size(); index++) {
            output.push_back(data_to_unpack[index] & 0xFFFF);
            output.push_back(data_to_unpack[index] >> 16);
        }
        return output;
    } else if constexpr (std::is_same_v<DataType, bfloat16>) {
        return unpack_uint32_vec_into_bfloat16_vec(data_to_unpack);
    } else if constexpr (std::is_same_v<DataType, float>) {
        union float_uint32_convert {
            uint32_t u;
            float f;
            float_uint32_convert() : u(0) {}
        };
        std::vector<float> float_data;
        for (auto i = 0; i < data_to_unpack.size(); i++) {
            float_uint32_convert a;
            a.u = data_to_unpack[i];
            float_data.push_back(a.f);
        }
        return float_data;
    } else {
        static_assert(tt::stl::concepts::always_false_v<DataType>, "Don't know how to unpack uint32 data generically!");
    }
}

template <typename T>
constexpr inline uint32_t element_size_bytes() {
    return sizeof(T);
}

template <typename T>
constexpr inline uint32_t packed_buffer_size_bytes(uint32_t volume_unpacked_data) {
    auto num_type_in_u32 = sizeof(uint32_t) / sizeof(T);
    return (volume_unpacked_data/num_type_in_u32) * sizeof(uint32_t);
}

// Specialization for float because it gets converted to bfloat16 before being packed
template <>
constexpr inline uint32_t packed_buffer_size_bytes<float>(uint32_t volume_unpacked_data) {
    auto num_type_in_u32 = sizeof(uint32_t) / sizeof(float);
    return (volume_unpacked_data / num_type_in_u32) * sizeof(uint32_t);
}

// ======================================================================================
//                                  Layout converters
// ======================================================================================
namespace detail {
static std::vector<uint32_t> to_4D_shape(const Shape& shape) {
    if (shape.rank() == 1) {
        return {1, 1, 1, shape[-1]};
    } else if (shape.rank() == 2) {
        return {1, 1, shape[-2], shape[-1]};
    } else if (shape.rank() == 3) {
        return {1, shape[-3], shape[-2], shape[-1]};
    } else if (shape.rank() == 4) {
        return {shape[-4], shape[-3], shape[-2], shape[-1]};
    } else {
        TT_THROW("Rank {} is not supported!", shape.rank());
    }
}
}  // namespace detail

template <typename T, template<typename> typename BufferType>
inline std::vector<T> convert_layout_row_major_to_tile(const Shape& shape, const BufferType<T>& data_to_convert) {
    TT_ASSERT(
        (shape[-2] % tt::constants::TILE_HEIGHT == 0 && shape[-1] % tt::constants::TILE_WIDTH == 0),
        "Unsupported shape for tensor conversion");
    auto shape_vec = detail::to_4D_shape(shape);
    return convert_layout(data_to_convert, shape_vec, TensorLayout::LIN_ROW_MAJOR, TensorLayout::TILED32_4FACES);
}

template <typename T, template<typename> typename BufferType>
inline std::vector<T> convert_layout_tile_to_row_major(const Shape& shape, const BufferType<T>& data_to_convert) {
    auto shape_vec = detail::to_4D_shape(shape);
    return convert_layout(data_to_convert, shape_vec, TensorLayout::TILED32_4FACES, TensorLayout::LIN_ROW_MAJOR);
}

// ======================================================================================
//                                         Print
// ======================================================================================
std::ostream& operator<<(std::ostream& os, const DataType& dtype);

namespace detail {

template <typename T>
inline void print_datum(std::ostream& ss, T datum) {
    ss << datum;
}

template <>
inline void print_datum(std::ostream& ss, bfloat16 datum) {
    ss << datum.to_float();
}

template <typename BufferType>
std::string to_string(const BufferType& buffer, DataType dtype) {
    std::stringstream ss;
    ss << "[ ";
    for (int i = 0; i < buffer.size(); i++) {
        print_datum(ss, buffer[i]);
        if (i < buffer.size() - 1) {
            ss << ", ";
        }
    }
    ss << " dtype=" <<  dtype << " ]" << std::endl;
    return ss.str();
}

template <typename BufferType>
std::string to_string_row_major_0D(const BufferType& buffer, const Shape& shape, DataType dtype) {

    std::stringstream ss;
    ss << "Tensor( ";
    print_datum(ss, buffer[0]);
    ss << ", dtype=" <<  dtype << " )" << std::endl;
    return ss.str();
}

// TODO: make these configurable
const Shape MAX_NUM_ELEMENTS_TO_PRINT = Shape({4, 4, 32, 32});

template <typename BufferType>
std::string to_string_row_major_1D(const BufferType& buffer, const Shape& shape, DataType dtype) {

    std::stringstream ss;
    ss << "Tensor([ ";
    for (auto x = 0; x < std::min(shape[-1], MAX_NUM_ELEMENTS_TO_PRINT[-1]); x++) {
        // data in row major order
        auto index = x;
        print_datum(ss, buffer[index]);
        if (x < shape[-1] - 1) {
            ss << ", ";
        }
    }
    if (shape[-1] > MAX_NUM_ELEMENTS_TO_PRINT[-1]) {
        ss << "...";
    }
    ss << "], dtype=" <<  dtype << " )" << std::endl;
    return ss.str();
}

template <typename BufferType>
std::string to_string_row_major_2D(const BufferType& buffer, const Shape& shape, DataType dtype) {

    std::stringstream ss;
    ss << "Tensor([ ";
    for (auto y = 0; y < std::min(shape[-2], MAX_NUM_ELEMENTS_TO_PRINT[-2]); y++) {
        if (y == 0)
            ss << "[";
        else
            ss << "    [";
        for (auto x = 0; x < std::min(shape[-1], MAX_NUM_ELEMENTS_TO_PRINT[-1]); x++) {
            // data in row major order
            auto index = x + y*shape[1];
            print_datum(ss, buffer[index]);
            if (x < shape[1] - 1) {
                ss << ", ";
            }
        }
        if (shape[-1] > MAX_NUM_ELEMENTS_TO_PRINT[-1]) {
            ss << "...";
        }
        if(y < shape[0] - 1)
            ss << "]," << std::endl;
        else
            ss << "]";
    }
    if (shape[-2] > MAX_NUM_ELEMENTS_TO_PRINT[-2]) {
        ss << "...";
    }
    ss << "], dtype=" <<  dtype << " )" << std::endl;
    return ss.str();
}

template <typename BufferType>
std::string to_string_row_major_3D(const BufferType& buffer, const Shape& shape, DataType dtype) {

    std::stringstream ss;
    ss << "Tensor([ ";
    for (auto z = 0; z < std::min(shape[-3], MAX_NUM_ELEMENTS_TO_PRINT[-3]); z++) {
        if (z == 0)
            ss << "[";
        else
            ss << "   [";
        for (auto y = 0; y < std::min(shape[-2], MAX_NUM_ELEMENTS_TO_PRINT[-2]); y++) {
            if (y == 0)
                ss << "[";
            else
                ss << "    [";
            for (auto x = 0; x < std::min(shape[-1], MAX_NUM_ELEMENTS_TO_PRINT[-1]); x++) {
                // data in row major order
                auto index = x + y*shape[2] + z*shape[1]*shape[2];
                print_datum(ss, buffer[index]);
                if (x < shape[2] - 1) {
                    ss << ", ";
                }
            }
            if (shape[-1] > MAX_NUM_ELEMENTS_TO_PRINT[-1]) {
                ss << "...";
            }
            if(y < shape[1] - 1)
                ss << "]," << std::endl;
            else
                ss << "]";
        }
        if (shape[-2] > MAX_NUM_ELEMENTS_TO_PRINT[-2]) {
            ss << "...";
        }
        if(z < shape[0] - 1)
            ss << "]," << std::endl << std::endl;
        else
            ss << "]";
    }
    if (shape[-3] > MAX_NUM_ELEMENTS_TO_PRINT[-3]) {
        ss << "...";
    }
    ss << "], dtype=" <<  dtype << " )" << std::endl;
    return ss.str();
}

template <typename BufferType>
std::string to_string_row_major_4D(const BufferType& buffer, const Shape& shape, DataType dtype) {

    std::stringstream ss;
    ss << "Tensor([ ";
    for (auto w = 0; w < std::min(shape[-4], MAX_NUM_ELEMENTS_TO_PRINT[-4]); w++) {
        if(w == 0)
            ss << "[";
        else
            ss << "  [";
        for (auto z = 0; z < std::min(shape[-3], MAX_NUM_ELEMENTS_TO_PRINT[-3]); z++) {
            if (z == 0)
                ss << "[";
            else
                ss << "   [";
            for (auto y = 0; y < std::min(shape[-2], MAX_NUM_ELEMENTS_TO_PRINT[-2]); y++) {
                if (y == 0)
                    ss << "[";
                else
                    ss << "    [";
                for (auto x = 0; x < std::min(shape[-1], MAX_NUM_ELEMENTS_TO_PRINT[-1]); x++) {
                    // data in row major order
                    auto index = x + y*shape[3] + z*shape[2]*shape[3] + w*shape[1]*shape[2]*shape[3];
                    print_datum(ss, buffer[index]);
                    if (x < shape[3] - 1) {
                        ss << ", ";
                    }
                }
                if (shape[-1] > MAX_NUM_ELEMENTS_TO_PRINT[-1]) {
                    ss << "...";
                }
                if(y < shape[2] - 1)
                    ss << "]," << std::endl;
                else
                    ss << "]";
            }
            if (shape[-2] > MAX_NUM_ELEMENTS_TO_PRINT[-2]) {
                ss << "...";
            }
            if (z < shape[1] - 1)
                ss << "]," << std::endl << std::endl;
            else
                ss << "]";
        }
        if (shape[-3] > MAX_NUM_ELEMENTS_TO_PRINT[-3]) {
            ss << "...";
        }
        if (w < shape[0] - 1)
            ss << "]," << std::endl << std::endl << std::endl;
        else
            ss << "]";
    }
    if (shape[-4] > MAX_NUM_ELEMENTS_TO_PRINT[-4]) {
        ss << "...";
    }
    ss << "], dtype=" << dtype << " )" << std::endl;
    return ss.str();
}

template <typename BufferType>
std::string to_string_row_major(const BufferType& buffer, const Shape& shape, DataType dtype) {
    if (shape.rank() == 0) {
        return to_string_row_major_0D(buffer, shape, dtype);
    }
    if (shape.rank() == 1) {
        return to_string_row_major_1D(buffer, shape, dtype);
    }
    else if (shape.rank() == 2) {
        return to_string_row_major_2D(buffer, shape, dtype);
    }
    else if (shape.rank() == 3) {
        return to_string_row_major_3D(buffer, shape, dtype);
    }
    else if (shape.rank() == 4) {
        return to_string_row_major_4D(buffer, shape, dtype);
    }
    else {
        TT_THROW("Cannot print tensor of rank {}", shape.rank());
    }
}

}  // namespace detail

// ======================================================================================
//                                      Validators
// ======================================================================================
void validate_on_device_dtype_and_layout(Device *device, DataType dtype, Layout layout);

// -----------------------------------------------------------------------------------------------------------------------------------------------
// ===============================================================================================================================================
//                                                              High Level APIs
// ===============================================================================================================================================
// -----------------------------------------------------------------------------------------------------------------------------------------------

// ======================================================================================
//                           Data reader, writer, and initializers
// ======================================================================================
DeviceBuffer allocate_buffer_on_device(
    uint32_t buffer_size_bytes,
    Device *device,
    const Shape& shape,
    DataType data_type,
    Layout layout,
    const MemoryConfig& memory_config,
    std::optional<ShardSpecBuffer> shard_spec = std::nullopt
);

template <typename T>
inline void read_data_from_device_buffer(CommandQueue &cq, DeviceBuffer device_buffer, void* host_buffer_data, bool blocking) {
    std::cout << "reading from: " << device_buffer -> address() << std::endl;
    EnqueueReadBuffer(cq, device_buffer, host_buffer_data, true);
}

template <typename T>
inline void read_data_from_device_buffer(DeviceBuffer device_buffer, vector<T>& host_buffer) {
    std::vector<uint32_t> host_buffer_uint32;
    ::detail::ReadFromBuffer(device_buffer, host_buffer_uint32);
    host_buffer = unpack_uint32_vec<T>(host_buffer_uint32);
}

template <typename T, template <typename> typename BufferType>
inline void write_data_to_device_buffer(CommandQueue & cq, const BufferType<T>& host_buffer, DeviceBuffer device_buffer, bool writing_owned_storage) {
    ZoneScoped;
    // TODO(arakhmati): can we use generators in this function to go from `data_to_write` to `uint32_data`?
    // And effectively get rid of any additional allocation
    if (writing_owned_storage) {
        // std::cout << "Writing owned" << std::endl;
        EnqueueWriteBuffer( cq, device_buffer, host_buffer.get_vec(), false);
        // std::cout << "Done Writing owned" << std::endl;
    }
    else {
        // std::cout << "Writing borrowed" << std::endl;
        auto init_cq_mode = cq.get_mode();
        cq.set_mode(CommandQueue::CommandQueueMode::PASSTHROUGH);
        EnqueueWriteBuffer(cq, device_buffer, host_buffer.data(), false);
        cq.set_mode(init_cq_mode);
        // std::cout << "Done Writing borrowed" << std::endl;
    }
}

template <typename T, template <typename> typename BufferType>
inline void write_data_to_device_buffer(const BufferType<T>& host_buffer, Buffer& device_buffer) {
    ZoneScoped;
    // TODO(arakhmati): can we use generators in this function to go from `data_to_write` to `uint32_data`?
    // And effectively get rid of any additional allocation

    auto uint32_data = pack_vec_into_uint32_vec<T>(host_buffer);
    ::detail::WriteToBuffer(device_buffer, uint32_data);
}

template <typename T, template<typename> typename BufferType>
inline DeviceBuffer initialize_data_on_device(const BufferType<T>& data_to_write, Device* device, const Shape& shape,
            DataType data_type, Layout layout, const MemoryConfig& memory_config, std::optional<ShardSpecBuffer> shard_spec, bool writing_owned_storage) {
    ZoneScoped;
    TT_ASSERT(device != nullptr);
    auto packed_size_in_bytes = packed_buffer_size_bytes<T>(data_to_write.size());

    auto device_buffer = allocate_buffer_on_device(packed_size_in_bytes, device, shape, data_type, layout, memory_config, shard_spec); // Make this a command as well
    const char *TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        write_data_to_device_buffer<T>(device->command_queue(), data_to_write, device_buffer, writing_owned_storage);
    } else {
        std::cout << "Writing data to device " << std::endl;
        write_data_to_device_buffer<T>(data_to_write, *device_buffer);
        std::cout << "done" << std::endl;
    }
    // std::cout << "Data written" << std::endl;
    return device_buffer;
}

template <typename T>
inline DeviceBuffer to_device_buffer(const Storage& storage, Device* device, const Shape& shape, DataType data_type, Layout layout, const MemoryConfig& memory_config, std::optional<ShardSpecBuffer> shard_spec) {

    return std::visit(
        [&device, &shape, &data_type, &layout, memory_config, shard_spec] (auto&& storage) -> DeviceBuffer {

            using StorageType = std::decay_t<decltype(storage)>;
            if(memory_config.is_sharded()){
                TT_ASSERT(shard_spec.has_value(), "If sharded must provide shard_spec");
            }
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                auto data_to_write = owned_buffer::get_as<T>(storage.buffer);
                bool writing_owned_storage = true;
                TT_ASSERT(
                    compute_buffer_size(shape, data_type) == data_to_write.size(),
                    fmt::format("Tensor buffer size and number of data elements does not match: {} != {}", compute_buffer_size(shape, data_type), data_to_write.size())
                );
                if (layout == Layout::TILE) {
                    TT_ASSERT(
                        (shape[-2] % tt::constants::TILE_HEIGHT == 0 && shape[-1] % tt::constants::TILE_WIDTH == 0),
                        "Tensor shape incompatible for specified layout");
                }
                return initialize_data_on_device<T>(data_to_write, device, shape, data_type, layout, memory_config,  shard_spec, writing_owned_storage);
            }
            else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_THROW("Device storage doesn't support to_device_buffer");
            }
            else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                if constexpr (
                    std::is_same_v<T, float> or std::is_same_v<T, bfloat16> or std::is_same_v<T, std::uint32_t> or
                    std::is_same_v<T, std::uint16_t>) {
                    auto data_to_write = borrowed_buffer::get_as<T>(storage.buffer);
                    bool writing_owned_storage = false;
                    TT_ASSERT(
                        compute_buffer_size(shape, data_type) == data_to_write.size(),
                        fmt::format("Tensor buffer size and number of data elements does not match: {} != {}", compute_buffer_size(shape, data_type), data_to_write.size())
                    );
                    if (layout == Layout::TILE) {
                        TT_ASSERT(
                            (shape[-2] % tt::constants::TILE_HEIGHT == 0 && shape[-1] % tt::constants::TILE_WIDTH == 0),
                            "Tensor shape incompatible for specified layout");
                    }
                    return initialize_data_on_device<T>(data_to_write, device, shape, data_type, layout, memory_config,  shard_spec, writing_owned_storage);
                } else {
                    TT_THROW("Borrowed storage doesn't support this data type");
                }
            }
            else {
                raise_unsupported_storage<StorageType>();
            }
        },
        storage
    );
}

// ======================================================================================
//                                         .to()
// ======================================================================================
template <typename T>
inline Tensor to_host(const Tensor &tensor, bool blocking = true) {
    if (tensor.storage_type() != StorageType::DEVICE) {
        return tensor;
    }
    TT_ASSERT(tensor.is_allocated(), "Buffer must be allocated on device!");
    auto device_buffer = tensor.device_buffer();
    auto device = tensor.device();
    TT_ASSERT(device != nullptr && "Need device to be set copy data from device to host!");
    uint32_t size_in_bytes = device_buffer->size();
    vector<T> data_vec;
    const char *TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        data_vec.resize(size_in_bytes / sizeof(T));
        // std::cout << "calling read from device" << std::endl;
        read_data_from_device_buffer<T>(device->command_queue(), device_buffer, data_vec.data(), blocking);
        // std::cout << "done" << std::endl;
    } else {
        read_data_from_device_buffer<T>(device_buffer, data_vec);
    }
    // std::cout << "Read done in main thread" << std::endl;
    auto output_buffer = owned_buffer::create<T>(std::move(data_vec));
    return Tensor(OwnedStorage{output_buffer}, tensor.shape(), tensor.dtype(), tensor.layout());
}

template <typename T>
inline Tensor to_host_sharded(const Tensor &tensor) {
    TT_ASSERT(tensor.is_allocated(), "Buffer must be allocated on device!");
    auto device_buffer = tensor.buffer();
    auto device = tensor.device();
    TT_ASSERT(device != nullptr && "Need device to be set copy data from device to host!");
    uint32_t size_in_bytes = device_buffer->size();
    std::vector<uint32_t> device_data;
    const char *TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        TT_THROW("FAST_DISPATCH is not supported for to_host_sharded!");
    }
    ::detail::ReadFromBuffer(*device_buffer, device_data, true);
    auto data_vec = unpack_uint32_vec<T>(device_data);
    auto output_buffer = owned_buffer::create<T>(std::move(data_vec));
    return Tensor(OwnedStorage{output_buffer}, tensor.shape(), tensor.dtype(), tensor.layout());
}


template <typename T>
inline Tensor to_device(const Tensor &tensor, Device *target_device, const MemoryConfig &memory_config) {
    TT_ASSERT(tensor.storage_type() != StorageType::DEVICE);
    if (tensor.storage_type() ==  StorageType::OWNED) {
        TT_ASSERT(tensor.is_allocated(), "Need host buffer on device to exist to copy data to device!");
    }
    TT_ASSERT(target_device != nullptr && "Need target device in order to move tensor to device!");
    TT_ASSERT(tensor.is_allocated() && "Need data to exist in order to move it to device");

    auto shape = tensor.shape();
    auto data_type = tensor.dtype();
    auto layout = tensor.layout();

    std::optional<ShardSpecBuffer> shard_spec_buffer_opt = std::nullopt;
    if(memory_config.is_sharded()){
        auto page_shape = get_sharded_page_shape(layout, shape, data_type, memory_config.shard_spec.value().num_cores(), memory_config.shard_spec.value().shape);
        std::array<uint32_t, 2> tensor2d_size = {shape[0]*shape[1] * shape[2]/ page_shape[0],
                                                shape[3]/page_shape[1]
                                            };
        shard_spec_buffer_opt = ShardSpecBuffer(memory_config.shard_spec.value(), page_shape, tensor2d_size);
    }

    auto device_buffer = tensor_impl::to_device_buffer<T>(
        tensor.storage(), target_device, shape,
        data_type, layout, memory_config,
        shard_spec_buffer_opt
    );
    return Tensor(DeviceStorage{device_buffer}, shape, data_type, layout);
}


template <typename T>
inline Tensor to_layout(const Tensor &tensor, Layout target_layout) {
    if(tensor.layout() == target_layout) {
        return tensor;
    }

    auto shape = tensor.shape();
    auto source_layout = tensor.layout();
    auto convert = [&shape, source_layout, target_layout](const auto& input_data) -> std::vector<T> {
        switch (source_layout) {
            case Layout::ROW_MAJOR:
                if (target_layout == Layout::TILE) {
                    return convert_layout_row_major_to_tile(shape, input_data);
                }
                else {
                    TT_THROW("Unsupported layout conversion");
                }
                break;
            case Layout::TILE:
                if (target_layout == Layout::ROW_MAJOR) {
                    return convert_layout_tile_to_row_major(shape, input_data);
                }
                else {
                    TT_THROW("Unsupported layout conversion");
                }
                break;
            default:
                TT_THROW("Unsupported layout conversion");
        }
    };

    auto output_data = std::visit(
        [&convert] (auto&& storage) -> std::vector<T> {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                const auto input_data = owned_buffer::get_as<T>(storage.buffer);
                return convert(input_data);
            }
            else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                const auto input_data = borrowed_buffer::get_as<T>(storage.buffer);
                return convert(input_data);
            }
            else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            }
            else {
                raise_unsupported_storage<StorageType>();
            }
        },
        tensor.storage()
    );

    auto output_buffer = owned_buffer::create<T>(std::move(output_data));
    return Tensor(OwnedStorage{output_buffer}, tensor.shape(), tensor.dtype(), target_layout);
}

Tensor to_layout_bfloat8_b(const Tensor &tensor, Layout target_layout);

// ======================================================================================
//                                  .pad() and .unpad()
// ======================================================================================
template <typename T>
inline Tensor pad(const Tensor &tensor, const Shape& output_tensor_shape, const Shape& input_tensor_start, float pad_value) {
    auto pad_value_ = static_cast<T>(pad_value);
    const auto input_tensor_shape = tensor.shape();
    const auto input_tensor_strides = tensor.strides();
    const auto input_tensor_data_type = tensor.dtype();

    auto pad =
        [&input_tensor_shape, &input_tensor_strides, &input_tensor_data_type, &output_tensor_shape, &input_tensor_start, &pad_value_]
        (const auto& input_buffer) {
        // Check if input tensor fits in output tensor given the input tensor start indices
        TT_ASSERT(input_tensor_shape[0] + input_tensor_start[0] <= output_tensor_shape[0]);
        TT_ASSERT(input_tensor_shape[1] + input_tensor_start[1] <= output_tensor_shape[1]);
        TT_ASSERT(input_tensor_shape[2] + input_tensor_start[2] <= output_tensor_shape[2]);
        TT_ASSERT(input_tensor_shape[3] + input_tensor_start[3] <= output_tensor_shape[3]);

        // Figure out pad size on each dim
        uint32_t pad_size[4][2] = {
            {input_tensor_start[0], output_tensor_shape[0] - input_tensor_shape[0] - input_tensor_start[0]},
            {input_tensor_start[1], output_tensor_shape[1] - input_tensor_shape[1] - input_tensor_start[1]},
            {input_tensor_start[2], output_tensor_shape[2] - input_tensor_shape[2] - input_tensor_start[2]},
            {input_tensor_start[3], output_tensor_shape[3] - input_tensor_shape[3] - input_tensor_start[3]}
        };

        const std::array<uint32_t, 4> output_tensor_strides = {
            output_tensor_shape[1] * output_tensor_shape[2] * output_tensor_shape[3],
            output_tensor_shape[2] * output_tensor_shape[3],
            output_tensor_shape[3],
            1
        };

        auto output_buffer = owned_buffer::create<T>(compute_volume(output_tensor_shape));
        auto output_index = 0;
        for(auto i = 0; i < pad_size[0][0] * output_tensor_strides[0]; i++) {
            output_buffer[output_index++] = pad_value_;
        }
        for(auto dim0 = 0; dim0 < input_tensor_shape[0]; dim0++) {
            for(auto i = 0; i < pad_size[1][0] * output_tensor_strides[1]; i++) {
                output_buffer[output_index++] = pad_value_;
            }
            for(auto dim1 = 0; dim1 < input_tensor_shape[1]; dim1++) {
                for(auto i = 0; i < pad_size[2][0] * output_tensor_strides[2]; i++) {
                    output_buffer[output_index++] = pad_value_;
                }
                for(auto dim2 = 0; dim2 < input_tensor_shape[2]; dim2++) {
                    for(auto i = 0; i < pad_size[3][0] * output_tensor_strides[3]; i++) {
                        output_buffer[output_index++] = pad_value_;
                    }
                    for(auto dim3 = 0; dim3 < input_tensor_shape[3]; dim3++) {
                        auto input_index = dim3 + input_tensor_strides[2] * dim2 + input_tensor_strides[1] * dim1 + input_tensor_strides[0] * dim0;
                        output_buffer[output_index++] = input_buffer[input_index];
                    }
                    for(auto i = 0; i < pad_size[3][1] * output_tensor_strides[3]; i++) {
                        output_buffer[output_index++] = pad_value_;
                    }
                }
                for(auto i = 0; i < pad_size[2][1] * output_tensor_strides[2]; i++) {
                    output_buffer[output_index++] = pad_value_;
                }
            }
            for(auto i = 0; i < pad_size[1][1] * output_tensor_strides[1]; i++) {
                output_buffer[output_index++] = pad_value_;
            }
        }
        for(auto i = 0; i < pad_size[0][1] * output_tensor_strides[0]; i++) {
            output_buffer[output_index++] = pad_value_;
        }
        return output_buffer;
    };

    auto output_buffer = std::visit(
        [&pad] (auto&& storage) -> owned_buffer::Buffer<T> {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                const auto input_data = owned_buffer::get_as<T>(storage.buffer);
                return pad(input_data);
            }
            else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                const auto input_data = borrowed_buffer::get_as<T>(storage.buffer);
                return pad(input_data);
            }
            else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            }
            else {
                raise_unsupported_storage<StorageType>();
            }
        },
        tensor.storage()
    );
    return Tensor(OwnedStorage{output_buffer}, output_tensor_shape, tensor.dtype(), tensor.layout());
}

Tensor pad_bfloat8_b(const Tensor &tensor, const Shape& output_tensor_shape, const Shape& input_tensor_start, float pad_value);

template <typename T>
inline Tensor unpad(const Tensor &tensor, const Shape& output_tensor_start, const Shape& output_tensor_end) {
    const auto input_tensor_shape = tensor.shape();
    const auto input_tensor_strides = tensor.strides();

    // Check if tensor start and end indices are within input tensor shape
    TT_ASSERT(output_tensor_start[0] < input_tensor_shape[0]);
    TT_ASSERT(output_tensor_end[0] < input_tensor_shape[0]);
    TT_ASSERT(output_tensor_start[1] < input_tensor_shape[1]);
    TT_ASSERT(output_tensor_end[1] < input_tensor_shape[1]);
    TT_ASSERT(output_tensor_start[2] < input_tensor_shape[2]);
    TT_ASSERT(output_tensor_end[2] < input_tensor_shape[2]);
    TT_ASSERT(output_tensor_start[3] < input_tensor_shape[3]);
    TT_ASSERT(output_tensor_end[3] < input_tensor_shape[3]);

    // Check if start shape is <= end shape
    TT_ASSERT(output_tensor_start[0] <= output_tensor_end[0]);
    TT_ASSERT(output_tensor_start[1] <= output_tensor_end[1]);
    TT_ASSERT(output_tensor_start[2] <= output_tensor_end[2]);
    TT_ASSERT(output_tensor_start[3] <= output_tensor_end[3]);

    // Figure out output tensor shape
    const Shape output_tensor_shape = {
        output_tensor_end[0] - output_tensor_start[0] + 1,
        output_tensor_end[1] - output_tensor_start[1] + 1,
        output_tensor_end[2] - output_tensor_start[2] + 1,
        output_tensor_end[3] - output_tensor_start[3] + 1,
    };

    auto unpad =
        [&input_tensor_shape, &input_tensor_strides, &output_tensor_shape, &output_tensor_start, &output_tensor_end](
            const auto& input_buffer) {
            auto output_buffer = owned_buffer::create<T>(compute_volume(output_tensor_shape));
            auto output_index = 0;
            for (auto dim0 = output_tensor_start[0]; dim0 <= output_tensor_end[0]; dim0++) {
                for (auto dim1 = output_tensor_start[1]; dim1 <= output_tensor_end[1]; dim1++) {
                    for (auto dim2 = output_tensor_start[2]; dim2 <= output_tensor_end[2]; dim2++) {
                        for (auto dim3 = output_tensor_start[3]; dim3 <= output_tensor_end[3]; dim3++) {
                            auto input_index = dim3 + input_tensor_strides[2] * dim2 + input_tensor_strides[1] * dim1 +
                                               input_tensor_strides[0] * dim0;
                            output_buffer[output_index++] = input_buffer[input_index];
                        }
                    }
                }
            }
            return output_buffer;
        };

    auto output_buffer = std::visit(
        [&unpad](auto&& storage) -> owned_buffer::Buffer<T> {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                const auto input_data = owned_buffer::get_as<T>(storage.buffer);
                return unpad(input_data);
            } else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                const auto input_data = borrowed_buffer::get_as<T>(storage.buffer);
                return unpad(input_data);
            } else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            } else {
                raise_unsupported_storage<StorageType>();
            }
        },
        tensor.storage());
    return Tensor(OwnedStorage{output_buffer}, output_tensor_shape, tensor.dtype(), tensor.layout());
}


Tensor unpad_bfloat8_b(const Tensor &tensor, const Shape& output_tensor_start, const Shape& output_tensor_end);

// ======================================================================================
//                                         Print
// ======================================================================================
template <typename T>
inline std::string to_string(const Tensor &tensor, Layout print_layout, bool pretty_print = false) {

    const auto shape = tensor.shape();
    const auto dtype = tensor.dtype();
    const auto layout = tensor.layout();

    auto to_string_impl = [&print_layout, &pretty_print, &shape, &dtype, &layout](const auto& buffer) -> std::string {
        switch (layout) {
            case Layout::ROW_MAJOR:
                if (print_layout == Layout::ROW_MAJOR) {
                    return pretty_print ? detail::to_string_row_major(buffer, shape, dtype) : detail::to_string(buffer, dtype);
                } else if (print_layout == Layout::TILE) {
                    TT_ASSERT(pretty_print == false && "Can only pretty print in Row Major layout!");
                    auto converted_data = convert_layout_row_major_to_tile(shape, buffer);
                    return detail::to_string(converted_data, dtype);
                }
                else {
                    TT_THROW("Unsupported print layout");
                }
                break;
            case Layout::TILE:
                if (print_layout == Layout::ROW_MAJOR) {
                    auto converted_data = convert_layout_tile_to_row_major(shape, buffer);
                    return pretty_print ? detail::to_string_row_major(converted_data, shape, dtype) : detail::to_string(converted_data, dtype);
                } else if (print_layout == Layout::TILE) {
                    TT_ASSERT(pretty_print == false && "Can only pretty print in Row Major layout!");
                    return detail::to_string(buffer, dtype);
                } else {
                    TT_THROW("Unsupported print layout");
                }
                break;
            default:
                TT_THROW("Unsupported print layout");
        }
    };

    return std::visit(
        [&] (auto&& storage) -> std::string {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                const auto input_data = owned_buffer::get_as<T>(storage.buffer);
                return to_string_impl(input_data);
            }
            else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                const auto input_data = borrowed_buffer::get_as<T>(storage.buffer);
                return to_string_impl(input_data);
            }
            else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                return to_string<T>(to_host<T>(tensor), print_layout, pretty_print);
            }
            else {
                raise_unsupported_storage<StorageType>();
            }
        },
        tensor.storage()
    );
}


template <typename T>
Tensor extract_shard(const Tensor & tensor, const uint32_t & core_id){

    auto buffer= tensor.buffer();
    auto buffer_shard_shape = buffer->shard_spec().shape();
    std::array <uint32_t, 4> shard_shape_array = {1,1,buffer_shard_shape[0],buffer_shard_shape[1]};
    Shape shard_shape(shard_shape_array);
    std::vector<uint32_t> device_data;
    ::detail::ReadShard(*buffer, device_data, core_id);


    auto unpacked_data = tensor_impl::unpack_uint32_vec<T>(device_data);
    auto output_buffer = owned_buffer::create<T>(std::move(unpacked_data));
    return Tensor(OwnedStorage{output_buffer}, shard_shape, tensor.dtype(), tensor.layout());

}

template <typename DataType>
void* get_raw_host_data_ptr(const Tensor& tensor) {
    return std::visit(
        [](auto&& storage) -> void* {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                auto buffer = owned_buffer::get_as<DataType>(storage.buffer);
                return buffer.data();
            } else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                if constexpr (
                    std::is_same_v<DataType, float> or std::is_same_v<DataType, bfloat16> or
                    std::is_same_v<DataType, std::uint32_t> or std::is_same_v<DataType, std::uint16_t>) {
                    auto buffer = borrowed_buffer::get_as<DataType>(storage.buffer);
                    return buffer.data();
                } else {
                    TT_THROW("Borrowed storage doesn't support this data type");
                }
            } else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            } else {
                raise_unsupported_storage<StorageType>();
            }
        },
        tensor.storage());
}

template <typename DataType>
void memcpy(Tensor& dst, const Tensor& src) {
    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE != nullptr) {
        TT_THROW("SLOW_DISPATCH is not supported for memcpy!");
    }

    TT_ASSERT(dst.dtype() == src.dtype());
    TT_ASSERT(dst.layout() == src.layout());

    if (is_cpu_tensor(dst) && is_device_tensor(src)) {
        EnqueueReadBuffer(
            src.device()->command_queue(), src.device_buffer(), get_raw_host_data_ptr(dst), true);
    } else if (is_device_tensor(dst) && is_cpu_tensor(src)) {
        EnqueueWriteBuffer(
            dst.device()->command_queue(), dst.device_buffer(), get_raw_host_data_ptr(src), false);
    } else {
        TT_THROW("Unsupported memcpy");
    }
}

}  // namespace tensor_impl

}  // namespace tt_metal

}  // namespace tt
