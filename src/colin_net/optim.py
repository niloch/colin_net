from abc import abstractclassmethod, abstractmethod
from enum import Enum
from functools import partial
from typing import Any, Callable, Tuple

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

    net: Model
    value_grad_func: LossGrad
    lr: float = 0.01

    @classmethod
    def initialize(cls, net: Model, loss: Loss, lr: float = 0.01) -> "SGD":
        value_grad_func = jit(value_and_grad(loss))

        return cls(net=net, value_grad_func=value_grad_func, lr=lr, loss=loss)

    def step(self, inputs: Tensor, targets: Tensor) -> Tuple[float, Model]:

        loss, grads = self.value_grad_func(self.net, inputs, targets)

        combiner = partial(sgd_update_combiner, lr=self.lr)
        self.net = tree_multimap(combiner, self.net, grads)
        return loss, self.net


class Adam(Optimizer):

    value_grad_func: LossGrad
    init_func: Callable
    update_func: Callable
    get_params: Callable
    opt_state: Any
    update_count: int
    lr: float = 0.01

    @classmethod
    def initialize(cls, net: Model, loss: Loss, lr: float = 0.01) -> "Adam":

        value_grad_func: LossGrad = jit(value_and_grad(loss))
        init_fun, update_fun, get_params = adam(step_size=lr)
        opt_state = init_fun(net)
        update_count = 0
        return cls(
            value_grad_func=value_grad_func,
            init_func=init_fun,
            update_func=update_fun,
            get_params=get_params,
            opt_state=opt_state,
            update_count=update_count,
            lr=lr,
            loss=loss,
        )

    def step(self, inputs: Tensor, targets: Tensor) -> Tuple[float, Model]:
        net = self.get_params(self.opt_state)
        loss, grads = self.value_grad_func(net, inputs, targets)
        self.opt_state = self.update_func(self.update_count, grads, self.opt_state)
        self.update_count += 1
        return loss, self.get_params(self.opt_state)


OPTIMIZERS = {"sgd": SGD, "adam": Adam}


class OptimizerEnum(str, Enum):
    sgd = "sgd"
    adam = "adam"
