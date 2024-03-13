import tt_lib.device
import tt_lib.tensor
from typing import List

class DeviceMesh:
    def __init__(self, *args, **kwargs) -> None: ...
    def get_device(self, arg0: int) -> tt_lib.device.Device: ...
    def get_device_ids(self) -> List[int]: ...
    def get_num_devices(self) -> int: ...

def aggregate_as_tensor(tensors: List[tt_lib.tensor.Tensor]) -> tt_lib.tensor.Tensor: ...
def close_device_mesh(device_mesh: DeviceMesh) -> None: ...
def from_device_mesh(tensor: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor: ...
def get_device_tensors(tensor: tt_lib.tensor.Tensor) -> List[tt_lib.tensor.Tensor]: ...
def open_device_mesh(*args, **kwargs): ...
def to_device_mesh(
    tensor: tt_lib.tensor.Tensor, device_mesh: DeviceMesh, memory_config: tt_lib.tensor.MemoryConfig
) -> tt_lib.tensor.Tensor: ...
