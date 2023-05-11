#include "frameworks/tt_dispatch/impl/command.hpp"
// #include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/common/base.hpp"

using namespace tt::tt_metal;

struct SystemMemoryCBWriteInterface {
    uint fifo_wr_ptr;
    uint fifo_limit;
    uint fifo_size; // Size in bytes of FIFO
    uint fifo_size_commands; // How many commands are in fifo
};

class SystemMemoryWriter {
    SystemMemoryCBWriteInterface cb_write_interface;

   public:
    SystemMemoryWriter() {}

    // Ensure that there is enough space to push to the queue first
    void cb_reserve_back(Device* device) {

        // TT_THROW("cb_reserve_back not implemented yet");
        uint* commands_received_ptr; // How many commands I sent
        uint* commands_acked_ptr; // How many commands were acknowledged

        uint commands_received; // = *commands_received_ptr;

        bool free_space;
        do {
            uint commands_acked;
            uint free_space_commands = this->cb_write_interface.fifo_size_commands - (commands_received - commands_acked);
            free_space = (bool)free_space_commands;
        } while (not free_space);
    }

    void noc_write(Device* device, const DeviceCommand& command) {
        const array<uint, SIZE>& desc = command.get_desc();
        vector<uint32_t> command_vector(desc.begin(), desc.end());
        device->cluster()->write_sysmem_vec(command_vector, cb_write_interface.fifo_wr_ptr, 0);
    }

    void cb_push_back(Device* device) {
        // Notify dispatch core
        this->cb_write_interface.fifo_wr_ptr += DeviceCommand::size() * sizeof(uint);

        if (this->cb_write_interface.fifo_wr_ptr > this->cb_write_interface.fifo_limit) {
            this->cb_write_interface.fifo_wr_ptr -= this->cb_write_interface.fifo_size;
        }
    }
};
