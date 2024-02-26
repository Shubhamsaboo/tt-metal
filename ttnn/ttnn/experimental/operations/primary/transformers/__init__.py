# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import inspect
import sys

import tt_lib as ttl
import ttnn


THIS_MODULE = sys.modules[__name__]

__all__ = []

for attribute_name in dir(ttl.operations.primary.transformers):
    if attribute_name.startswith("__"):
        continue
    attribute = getattr(ttl.operations.primary.transformers, attribute_name)
    if inspect.isbuiltin(attribute) and (
        "tt_lib.tensor.Tensor" in attribute.__doc__ or "tt::tt_metal::Tensor" in attribute.__doc__
    ):
        attribute = ttnn.decorators.register_ttl_operation_as_ttnn_operation(
            name=f"ttnn.experimental.operations.primary.transformers.{attribute_name}", function=attribute
        )
    setattr(THIS_MODULE, attribute_name, attribute)
    __all__.append(attribute_name)
