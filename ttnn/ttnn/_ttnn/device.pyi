import tt_lib.device

def close_device(device: tt_lib.device.Device) -> None: ...
def disable_and_clear_program_cache(device: tt_lib.device.Device) -> None: ...
def enable_program_cache(device: tt_lib.device.Device) -> None: ...
def open_device(*args, **kwargs): ...
