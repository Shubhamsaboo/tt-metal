// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "common/bfloat16.hpp"
#include "tensor/borrowed_buffer.hpp"
#include "tensor/owned_buffer.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/tt_stl/concepts.hpp"
#include "tt_metal/tt_stl/reflection.hpp"

namespace tt {

namespace tt_metal {

enum class Layout {
    ROW_MAJOR = 0,
    TILE = 1
};

enum class DataType {
    BFLOAT16 = 0,
    FLOAT32 = 1,
    UINT32 = 2,
    BFLOAT8_B = 3,
    UINT16 = 4,
};

enum class StorageType {
    OWNED,
    DEVICE,
    BORROWED,  // for storing torch/numpy/etc tensors
};



tt::DataFormat datatype_to_dataformat_converter(DataType datatype);


static constexpr std::size_t MAX_NUM_DIMENSIONS = 8;

struct Padding {
    enum class PadValue {
        Any,
        Zero,
        Infinity,
        NegativeInfinity
    };

    struct PadDimension {
        std::size_t front;
        std::size_t back;

        static constexpr auto attribute_names = std::make_tuple("front", "back");
        const auto attribute_values() const { return std::make_tuple(std::cref(this->front), std::cref(this->back)); }
    };

    std::size_t rank_;
    std::array<PadDimension, MAX_NUM_DIMENSIONS> pad_dimensions_;
    PadValue pad_value_;

    Padding(const Padding&) = default;
    Padding& operator=(const Padding&) = default;
    Padding(Padding&&) = default;
    Padding& operator=(Padding&&) = default;
    ~Padding() = default;

    Padding(const std::size_t rank);
    Padding(const std::initializer_list<PadDimension> pad_dimensions, PadValue pad_value);
    Padding(const std::vector<PadDimension>& pad_dimensions, PadValue pad_value);

    template <std::size_t Rank>
    Padding(const std::array<std::array<uint32_t, 2>, Rank> pad_dimensions, PadValue pad_value) :
        rank_(pad_dimensions.size()), pad_dimensions_{}, pad_value_(pad_value) {
        for (auto index = 0; index < Rank; index++) {
            this->pad_dimensions_[index] = {.front = pad_dimensions[index][0], .back = pad_dimensions[index][1]};
        }
    }

    PadDimension& operator[](const std::int64_t index);
    const PadDimension& operator[](const std::int64_t index) const;

    PadValue pad_value() const;

    static constexpr auto attribute_names = std::make_tuple("rank", "pad_dimensions", "pad_value");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->rank_), std::cref(this->pad_dimensions_), std::cref(this->pad_value_));
    }
    friend std::ostream& operator<<(std::ostream& os, const Padding& padding);
};

inline std::ostream& operator<<(std::ostream& os, const Padding& padding) {
    os << "Padding(";
    os << "rank: " << padding.rank_;
    os << ", pad_dimensions: [";
    for (std::size_t i = 0; i < padding.rank_; ++i) {
        os << "{front: " << padding.pad_dimensions_[i].front << ", back: " << padding.pad_dimensions_[i].back << "}";
        if (i < padding.rank_ - 1) os << ", ";
    }
    os << "]";
    os << ", pad_value: ";
    switch (padding.pad_value_) {
        case Padding::PadValue::Any: os << "Any"; break;
        case Padding::PadValue::Zero: os << "Zero"; break;
        case Padding::PadValue::Infinity: os << "Infinity"; break;
        case Padding::PadValue::NegativeInfinity: os << "NegativeInfinity"; break;
        default: os << "Unknown";
    }
    os << ")";
    return os;
}

bool operator==(const Padding&, const Padding&);
bool operator!=(const Padding&, const Padding&);

class Shape {
    std::size_t rank_;
    std::array<uint32_t, MAX_NUM_DIMENSIONS> dimensions_;
    Padding padding_;

   public:
    Shape(const Shape&) = default;
    Shape& operator=(const Shape&) = default;
    Shape(Shape&&) = default;
    Shape& operator=(Shape&&) = default;
    ~Shape() = default;

    Shape(const std::initializer_list<uint32_t>);
    Shape(const std::vector<uint32_t>&);
    Shape(const std::initializer_list<uint32_t>, const Padding&);
    Shape(const std::vector<uint32_t>&, const Padding&);

    explicit Shape(const Shape&, const Padding&);

    template <std::size_t Rank>
    Shape(const std::array<uint32_t, Rank> &shape) : rank_(Rank), dimensions_{}, padding_{Rank} {
        for (auto index = 0; index < Rank; index++) {
            this->dimensions_[index] = shape[index];
        }
    }

    template <std::size_t Rank>
    explicit Shape(const std::array<uint32_t, Rank> &shape, const std::array<uint32_t, Rank> &padded_shape) :
        rank_(Rank), dimensions_{}, padding_{Rank} {
        for (auto index = 0; index < Rank; index++) {
            auto padded_dimension = padded_shape[index];
            this->dimensions_[index] = padded_dimension;
            this->padding_[index] = {.front = 0, .back = padded_dimension - shape[index]};
        }
    }

    std::size_t rank() const;

    uint32_t& operator[](const std::int64_t index);
    const uint32_t& operator[](const std::int64_t index) const;

    const uint32_t* begin() const;
    const uint32_t* end() const;

    const Padding& padding() const;
    const Shape without_padding() const;

    const uint32_t get_normalized_index(std::int64_t index) const;

    static constexpr auto attribute_names = std::make_tuple("rank", "dimensions", "padding");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->rank_), std::cref(this->dimensions_), std::cref(this->padding_));
    }
    friend std::ostream& operator<<(std::ostream& os, const Shape& shape);
};

inline std::ostream& operator<<(std::ostream& os, const Shape& padded_shape) {
    os << "Shape([";
    const auto shape = padded_shape.without_padding();
    const auto& padding = padded_shape.padding();
    for (auto i = 0; i < shape.rank(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        if (padding[i].front > 0) {
            os << padding[i].front << " + ";
        }
        os << shape[i];
        if (padding[i].back > 0) {
            os << " + " << padding[i].back;
        }
    }
    os << "])";
    return os;
}


bool operator==(const Shape&, const Shape&);
bool operator!=(const Shape&, const Shape&);

struct MemoryConfig {
    TensorMemoryLayout memory_layout = TensorMemoryLayout::INTERLEAVED;    // Interleave the data across multiple banks
    BufferType buffer_type = BufferType::DRAM; // Can be either DRAM or L1
    std::optional<ShardSpec> shard_spec = std::nullopt;
    bool is_sharded() const;

    static constexpr auto attribute_names = std::make_tuple("memory_layout", "buffer_type", "shard_spec");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->memory_layout), std::cref(this->buffer_type), std::cref(this->shard_spec));
    }
};

bool operator==(const MemoryConfig& config_a, const MemoryConfig& config_b);
bool operator!=(const MemoryConfig& config_a, const MemoryConfig& config_b);

using OwnedBuffer = std::variant<
    owned_buffer::Buffer<uint16_t>,
    owned_buffer::Buffer<uint32_t>,
    owned_buffer::Buffer<float>,
    owned_buffer::Buffer<bfloat16>>;
struct OwnedStorage {
    OwnedBuffer buffer;

    static constexpr auto attribute_names = std::make_tuple();
    const auto attribute_values() const { return std::make_tuple(); }
};

using DeviceBuffer = std::shared_ptr<Buffer>;
struct DeviceStorage {
    DeviceBuffer buffer;

    const MemoryConfig memory_config() const {
        if (this->buffer.get() == nullptr) {
            TT_THROW("MemoryConfig can only be obtained if the buffer is not null");
        }

        std::optional<ShardSpec> shard_spec = std::nullopt;
        if (is_sharded(this->buffer->buffer_layout())) {
            shard_spec = this->buffer->shard_spec().tensor_shard_spec;
        }
        return MemoryConfig{
            .memory_layout = this->buffer->buffer_layout(),
            .buffer_type = this->buffer->buffer_type(),
            .shard_spec = shard_spec};
    }

    static constexpr auto attribute_names = std::make_tuple("memory_config");
    const auto attribute_values() const { return std::make_tuple(this->memory_config()); }
};

using BorrowedBuffer = std::variant<
    borrowed_buffer::Buffer<uint16_t>,
    borrowed_buffer::Buffer<uint32_t>,
    borrowed_buffer::Buffer<float>,
    borrowed_buffer::Buffer<bfloat16>>;
struct BorrowedStorage {
    BorrowedBuffer buffer;
    std::function<void()> on_creation_callback = []{};
    std::function<void()> on_destruction_callback = []{};


    explicit BorrowedStorage(const BorrowedBuffer& buffer, const std::function<void()>& on_creation_callback, const std::function<void()>& on_destruction_callback)
    : buffer(buffer), on_creation_callback(on_creation_callback), on_destruction_callback(on_destruction_callback) {
        this->on_creation_callback();
    }

    BorrowedStorage(const BorrowedStorage& other)
    : buffer(other.buffer), on_creation_callback(other.on_creation_callback), on_destruction_callback(other.on_destruction_callback) {
        this->on_creation_callback();
    }

    BorrowedStorage operator=(const BorrowedStorage& other) {
        this->buffer = other.buffer;
        this->on_creation_callback = other.on_creation_callback;
        this->on_destruction_callback = other.on_destruction_callback;
        this->on_creation_callback();
        return *this;
    }

    BorrowedStorage(BorrowedStorage&& other)
    : buffer(other.buffer), on_creation_callback(other.on_creation_callback), on_destruction_callback(other.on_destruction_callback) {
        other.on_creation_callback = []{};
        other.on_destruction_callback = []{};
    }

    BorrowedStorage operator=(BorrowedStorage&& other) {
        this->buffer = other.buffer;
        this->on_creation_callback = other.on_creation_callback;
        this->on_destruction_callback = other.on_destruction_callback;
        other.on_creation_callback = []{};
        other.on_destruction_callback = []{};
        return *this;
    }

    ~BorrowedStorage() {
        this->on_destruction_callback();
    }

    static constexpr auto attribute_names = std::make_tuple();
    const auto attribute_values() const { return std::make_tuple(); }
};

using Storage = std::variant<
    OwnedStorage,
    DeviceStorage,
    BorrowedStorage
>;

template<typename T>
constexpr void raise_unsupported_storage() {
    static_assert(tt::stl::concepts::always_false_v<T>, "Unsupported Storage");
}

inline bool operator==(const Storage& v1, const Storage& v2) {
    return std::visit([](const auto& a, const auto& b) -> bool {
        if constexpr (std::is_same_v<decltype(a), decltype(b)>) {
            return a == b;
        } else {
            return false;
        }
    }, v1, v2);
};

}  // namespace tt_metal

}  // namespace tt

namespace ttnn {
namespace types {

namespace detail {
template <std::size_t Rank>
static tt::tt_metal::Shape compute_ttl_shape(
    const std::array<uint32_t, Rank> &shape, const std::array<std::array<uint32_t, 2>, Rank> &padding) {
    auto ttl_shape = std::array<uint32_t, Rank>{};
    for (auto index = 0; index < Rank; index++) {
        ttl_shape[index] = shape[index] + padding[index][0] + padding[index][1];
    }
    return tt::tt_metal::Shape{
        tt::tt_metal::Shape{ttl_shape}, tt::tt_metal::Padding{padding, tt::tt_metal::Padding::PadValue::Any}};
}

}  // namespace detail

template <std::size_t Rank>
struct RankedShape {
    const std::size_t rank;
    const tt::tt_metal::Shape value;

    explicit RankedShape(tt::tt_metal::Shape &&shape) : rank{Rank}, value(shape) {}
    explicit RankedShape(const tt::tt_metal::Shape &shape) : rank{Rank}, value(shape) {}

    explicit RankedShape(const std::array<uint32_t, Rank> &shape) : rank{Rank}, value{shape} {}

    explicit RankedShape(const std::array<uint32_t, Rank> &shape, const std::array<uint32_t, Rank> &padded_shape) :
        rank{Rank}, value{shape, padded_shape} {}

    explicit RankedShape(
        const std::array<uint32_t, Rank> &shape, const std::array<std::array<uint32_t, 2>, Rank> &padding) :
        rank{Rank}, value{detail::compute_ttl_shape(shape, padding)} {}

    RankedShape<Rank> padded() const {
        return RankedShape{tt::tt_metal::Shape{this->value, tt::tt_metal::Padding{this->value.rank()}}};
    }

    RankedShape<Rank> operator+(const std::array<std::array<uint32_t, 2>, Rank> &padding) const {
        auto shape = this->value;
        const auto &current_padding = this->value.padding();
        auto accumulated_padding = padding;
        for (auto index = 0; index < Rank; index++) {
            shape[index] += padding[index][0] + padding[index][1];
            accumulated_padding[index][0] += current_padding[index].front;
            accumulated_padding[index][1] += current_padding[index].back;
        }
        return RankedShape<Rank>{tt::tt_metal::Shape{
            shape, tt::tt_metal::Padding{accumulated_padding, tt::tt_metal::Padding::PadValue::Any}}};
    }

    template <std::size_t OtherRank>
    RankedShape<Rank> operator+(const std::array<std::array<uint32_t, 2>, OtherRank> &padding) const {
        TT_THROW("Invalid padding");
    }

    bool operator==(const RankedShape<Rank> &other) const { return this->value == other.value; }

    template <std::size_t OtherRank>
    bool operator==(const RankedShape<OtherRank> &other) const {
        return false;
    }

    const auto &operator[](std::int64_t index) const { return this->value.without_padding()[index]; }
};

template <std::size_t Rank>
static std::ostream &operator<<(std::ostream &os, const RankedShape<Rank> &self) {
    os << "ttnn.Shape([";
    const auto shape = self.value.without_padding();
    const auto &padding = self.value.padding();
    const auto &padded_shape = self.value;
    for (auto i = 0; i < Rank; ++i) {
        if (i > 0) {
            os << ", ";
        }
        if (padding[i].front > 0) {
            os << padding[i].front << " + ";
        }
        os << shape[i];
        if (padding[i].back > 0) {
            os << " + " << padding[i].back;
        }
    }
    os << "])";
    return os;
}

struct Shape {
    using RankedShapeVariant = std::variant<
        const RankedShape<1>,
        const RankedShape<2>,
        const RankedShape<3>,
        const RankedShape<4>,
        const RankedShape<5>,
        const RankedShape<6>,
        const RankedShape<7>,
        const RankedShape<8>>;

    const RankedShapeVariant ranked_shape;

   private:
    RankedShapeVariant ttl_shape_to_ttnn_shape(const tt::tt_metal::Shape &shape) {
        switch (shape.rank()) {
            case 1: return RankedShape<1>{shape};
            case 2: return RankedShape<2>{shape};
            case 3: return RankedShape<3>{shape};
            case 4: return RankedShape<4>{shape};
            case 5: return RankedShape<5>{shape};
            case 6: return RankedShape<6>{shape};
            case 7: return RankedShape<7>{shape};
            case 8: return RankedShape<8>{shape};
        };
        TT_THROW("Unsupported rank");
    }

   public:
    explicit Shape(const tt::tt_metal::Shape &shape) : ranked_shape{ttl_shape_to_ttnn_shape(shape)} {}

    template <std::size_t Rank>
    explicit Shape(const RankedShape<Rank> &shape) : ranked_shape{shape} {}

    template <std::size_t Rank>
    explicit Shape(const std::array<uint32_t, Rank> &shape) : ranked_shape{RankedShape<Rank>{shape}} {}

    template <std::size_t Rank>
    explicit Shape(const std::array<uint32_t, Rank> &shape, const std::array<uint32_t, Rank> &padded_shape) :
        ranked_shape{RankedShape<Rank>{shape, padded_shape}} {}

    template <std::size_t Rank>
    explicit Shape(const std::array<uint32_t, Rank> &shape, const std::array<std::array<uint32_t, 2>, Rank> &padding) :
        ranked_shape{RankedShape<Rank>{shape, padding}} {}

    const auto rank() const {
        return std::visit(
            []<std::size_t Rank>(const RankedShape<Rank> &shape) -> const auto { return Rank; }, this->ranked_shape);
    }

    Shape padded() const {
        return std::visit([](const auto &shape) -> Shape { return Shape(shape.padded()); }, this->ranked_shape);
    }

    template <std::size_t Rank>
    Shape operator+(const std::array<std::array<uint32_t, 2>, Rank> &padding) const {
        return std::visit(
            [&padding](const auto &shape) -> Shape { return Shape(shape + padding); }, this->ranked_shape);
    }

    bool operator==(const Shape &other) const {
        return std::visit(
            [](const auto &shape, const auto &other) -> bool { return shape == other; },
            this->ranked_shape,
            other.ranked_shape);
    }

    const auto &operator[](std::int64_t index) const {
        return std::visit([index](const auto &shape) -> decltype(auto) { return shape[index]; }, this->ranked_shape);
    }

    const auto &value() const {
        return std::visit([](const auto &shape) -> const auto & { return shape.value; }, this->ranked_shape);
    }

    template <std::size_t NewRank>
    const Shape to_rank() const {
        return std::visit(
            []<std::size_t Rank>(const RankedShape<Rank> &shape) {
                if constexpr (Rank == NewRank) {
                    return Shape(shape);
                } else {
                    auto num_missing_dims = NewRank - Rank;

                    std::array<uint32_t, NewRank> new_shape{};
                    std::array<uint32_t, NewRank> new_padded_shape{};

                    new_shape.fill(1);
                    new_padded_shape.fill(1);

                    for (auto index = 0; index < Rank; index++) {
                        new_shape[index + num_missing_dims] = shape[index];
                        new_padded_shape[index + num_missing_dims] = shape.padded()[index];
                    }
                    return Shape(RankedShape<NewRank>(new_shape, new_padded_shape));
                }
            },
            this->ranked_shape);
    }
};

static std::ostream &operator<<(std::ostream &os, const Shape &self) {
    std::visit([&os](const auto &shape) { os << shape; }, self.ranked_shape);
    return os;
}

}  // namespace types

using types::Shape;

}  // namespace ttnn
