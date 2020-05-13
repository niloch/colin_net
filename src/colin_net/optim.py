from enum import Enum
from typing import Any, Callable, List, Tuple

from jax import grad, jit
from jax.experimental.optimizers import adam
from jax.tree_util import tree_multimap

from colin_net.base import PyTreeLike
from colin_net.loss import Loss
from colin_net.nn import NeuralNet
from colin_net.tensor import Tensor

LossGrad = Loss
AdamState = Tuple[Callable, Callable, Callable, Callable, Any, int]


class Optimizer(PyTreeLike, is_abstract=True):
    def step(self, keys: Tensor, inputs: Tensor, targets: Tensor) -> NeuralNet:
        raise NotImplementedError

    @classmethod
    def initialize(cls, net: NeuralNet, loss: Loss, lr: float = 0.01) -> "Optimizer":
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, net: NeuralNet, grad_fun: LossGrad, lr: float) -> None:
        self.net = net
        self.grad_fun = grad_fun
        self.lr = lr

    @jit
    def sgd_update_combiner(self, param: Tensor, grad: Tensor) -> Tensor:
        """Convenvience method for performing SGD on custom jax Pytree objects"""
        return param - (self.lr * grad)

    @jit
    def step(self, keys: Tensor, inputs: Tensor, targets: Tensor) -> NeuralNet:
        grads = self.grad_fun(self.net, keys, inputs, targets)

        self.net = tree_multimap(self.sgd_update_combiner, self.net, grads)
        return self.net

    @classmethod
    def initialize(cls, net: NeuralNet, loss: Loss, lr: float = 0.01) -> "SGD":
        return cls(net, grad(loss), lr)

    def tree_flatten(self,) -> Tuple[List[None], Tuple[NeuralNet, Callable, float]]:
        return (
            [None],
            (self.net, self.grad_fun, self.lr),
        )

    @classmethod
    def tree_unflatten(
        cls, aux: Tuple[NeuralNet, Callable, float], params: Any
    ) -> "SGD":
        return cls(*aux)


class Adam(Optimizer):
    def __init__(
        self,
        grad_fun: LossGrad,
        init_fun: Callable,
        update_fun: Callable,
        get_params: Callable[..., NeuralNet],
        opt_state: Any,
        update_count: int,
    ) -> None:
        self.grad_fun = grad_fun
        self.init_fun = init_fun
        self.update_fun = update_fun
        self.get_params = get_params
        self.opt_state = opt_state
        self.update_count = update_count

    @classmethod
    def initialize(
        cls, net: NeuralNet, loss: Callable[..., Any], lr: float = 0.01
    ) -> "Adam":

        grad_fun = grad(loss)
        init_fun, update_fun, get_params = adam(step_size=lr)
        opt_state = init_fun(net)
        update_count = 0
        return cls(grad_fun, init_fun, update_fun, get_params, opt_state, update_count)

    @jit
    def step(self, keys: Tensor, inputs: Tensor, targets: Tensor) -> NeuralNet:
        net = self.get_params(self.opt_state)
        grads = self.grad_fun(net, keys, inputs, targets)
        self.opt_state = self.update_fun(self.update_count, grads, self.opt_state)
        self.update_count += 1
        return self.get_params(self.opt_state)

    def tree_flatten(self,) -> Tuple[List[None], AdamState]:
        return (
            [None],
            (
                self.grad_fun,
                self.init_fun,
                self.update_fun,
                self.get_params,
                self.opt_state,
                self.update_count,
            ),
        )

    @classmethod
    def tree_unflatten(cls, aux: AdamState, params: Any) -> "Adam":
        return cls(*aux)


OPTIMIZERS = {"sgd": SGD, "adam": Adam}


class OptimizerEnum(str, Enum):
    sgd = "sgd"
    adam = "adam"
