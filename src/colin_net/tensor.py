"""
A tensor is just a n-dimensional array
"""
from typing import Callable, Dict, Generator

import numpy as np
from jax.interpreters.xla import DeviceArray


class Tensor(DeviceArray):
    @classmethod
    def __get_validators__(cls) -> Generator[Callable, None, None]:
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema: Dict) -> None:
        # __modify_schema__ should mutate the dict it receives in place,
        # the returned value will be ignored
        field_schema.update(Tensor="jax.ndarray like")

    @classmethod
    def validate(cls, v: "Tensor") -> "Tensor":
        if not (isinstance(v, DeviceArray) or isinstance(v, np.ndarray)):
            raise TypeError("Tensor (jax.ndarray) required")
        return v


__all__ = ["Tensor"]
