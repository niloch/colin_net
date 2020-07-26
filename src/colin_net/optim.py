from abc import abstractclassmethod, abstractmethod
from enum import Enum
from functools import partial
from typing import Any, Callable, Optional, Tuple

from jax import jit, value_and_grad
from jax.experimental.optimizers import adam
from jax.tree_util import tree_multimap
from pydantic import BaseModel

from colin_net.loss import Loss
from colin_net.nn import Model
from colin_net.tensor import Tensor

LossGrad = Callable[[Model, Tensor, Tensor], Tuple[float, Model]]


class Optimizer(BaseModel):
    loss: Loss
    model: Model
    grads: Optional[Model] = None
    value_grad_func: LossGrad
    learning_rate: float = 0.001

    @abstractmethod
    def step(self, inputs: Tensor, targets: Tensor) -> Tuple[float, Model]:
        raise NotImplementedError

    @abstractclassmethod
    def initialize(cls, net: Model, loss: Loss, lr: float = 0.01) -> "Optimizer":
        raise NotImplementedError


@jit
def sgd_update_combiner(param: Tensor, grad: Tensor, lr: float) -> Tensor:
    """Convenvience method for performing SGD on custom jax Pytree objects"""
    return param - (lr * grad)


class SGD(Optimizer):
    @classmethod
    def initialize(cls, model: Model, loss: Loss, learning_rate: float = 0.01) -> "SGD":
        value_grad_func = jit(value_and_grad(loss))

        return cls(
            model=model,
            value_grad_func=value_grad_func,
            learning_rate=learning_rate,
            loss=loss,
        )

    def step(self, inputs: Tensor, targets: Tensor) -> Tuple[float, Model]:

        loss, self.grads = self.value_grad_func(self.model, inputs, targets)

        combiner = partial(sgd_update_combiner, lr=self.learning_rate)
        self.model = tree_multimap(combiner, self.model, self.grads)
        return loss, self.model


class Adam(Optimizer):
    init_func: Callable
    update_func: Callable
    get_params: Callable
    opt_state: Any
    update_count: int = 0

    @classmethod
    def initialize(
        cls, model: Model, loss: Loss, learning_rate: float = 0.01
    ) -> "Adam":

        value_grad_func: LossGrad = jit(value_and_grad(loss))
        init_fun, update_fun, get_params = adam(step_size=learning_rate)
        opt_state = init_fun(model)
        return cls(
            model=model,
            value_grad_func=value_grad_func,
            init_func=init_fun,
            update_func=update_fun,
            get_params=get_params,
            opt_state=opt_state,
            learning_rate=learning_rate,
            loss=loss,
        )

    def step(self, inputs: Tensor, targets: Tensor) -> Tuple[float, Model]:
        self.model = self.get_params(self.opt_state)
        loss, self.grads = self.value_grad_func(self.model, inputs, targets)
        self.opt_state = self.update_func(self.update_count, self.grads, self.opt_state)
        self.update_count += 1
        return loss, self.get_params(self.opt_state)


OPTIMIZERS = {"sgd": SGD, "adam": Adam}


class OptimizerEnum(str, Enum):
    sgd = "sgd"
    adam = "adam"
