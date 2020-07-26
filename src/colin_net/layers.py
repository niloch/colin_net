"""
Our neural nets will be made up of layers.
Each layer needs to pass the inputs forward. For example,
a neural net might look like

inputs -> Linear -> Tanh -> Linear -> output
"""
from abc import abstractmethod
from enum import Enum
from typing import Tuple

import jax.numpy as np
from jax import jit, nn, ops, random

from colin_net.base import Module, RNGWrapper
from colin_net.tensor import Tensor


class Mode(str, Enum):
    """Allowed values for Dropout Mode"""

    train = "train"
    eval = "eval"


INITIALIZERS = {
    "normal": nn.initializers.normal(stddev=1.0),
    "glorot_normal": nn.initializers.glorot_normal(),
    "lecun_normal": nn.initializers.lecun_normal(),
}


class InitializerEnum(str, Enum):
    normal = "normal"
    glorot_normal = "glorot_normal"
    lecun_normal = "lecun_normal"


class Layer(Module, is_abstract=True):
    """Abstract Class for Layers. Enforces subclasses to implement __call__"""

    @abstractmethod
    def __call__(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError


class ActivationEnum(str, Enum):
    tanh = "tanh"
    relu = "relu"
    leaky_relu = "leaky_relu"
    selu = "selu"
    sigmoid = "sigmoid"
    softmax = "softmax"
    mish = "mish"
    identity = "identity"


ACTIVATIONS = {
    "tanh": np.tanh,
    "relu": nn.relu,
    "leaky_relu": nn.leaky_relu,
    "selu": nn.selu,
    "sigmoid": nn.sigmoid,
    "softmax": nn.softmax,
    "mish": lambda x: x * np.tanh(nn.softplus(x)),
    "identity": lambda x: x,
}


class Linear(Layer):
    """Dense Linear Layer.
    Computes output = activation(np.dot(w, inputs) + b)"""

    w: Tensor
    b: Tensor
    activation: ActivationEnum = ActivationEnum.identity

    @jit
    def __call__(self, inputs: Tensor) -> Tensor:
        """outputs = np.dot(w, inputs) + b in single instance notation."""

        return ACTIVATIONS[self.activation](np.dot(self.w, inputs) + self.b)

    @classmethod
    def initialize(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        key: Tensor,
        activation: ActivationEnum = ActivationEnum.identity,
        initializer: InitializerEnum = InitializerEnum.normal,
    ) -> "Linear":
        """Factory for new Linear from input and output dimensions"""
        if initializer not in InitializerEnum.__members__:
            raise ValueError(
                f"initializer: {initializer} not in {InitializerEnum.__members__.values()}"
            )
        if activation not in ActivationEnum.__members__:
            raise ValueError(
                f"Activation: {activation} not in {ActivationEnum.__members__.values()}"
            )
        return cls(
            w=INITIALIZERS[initializer](key, shape=(output_dim, input_dim)),
            b=np.zeros(shape=(output_dim,)),
            activation=activation,
        )

    def tree_flatten(self) -> Tuple[Tuple[Tensor, Tensor], Tuple[ActivationEnum]]:
        return (self.w, self.b), (self.activation,)

    @classmethod
    def tree_unflatten(
        cls, aux: Tuple[ActivationEnum], params: Tuple[Tensor, Tensor]
    ) -> "Linear":
        return cls.construct(w=params[0], b=params[1], activation=aux[0])


class FrozenLinear(Linear):
    """Unrainable Linear Layer"""

    def tree_flatten(self) -> Tuple[Tuple[None], Tuple[Tensor, Tensor, ActivationEnum]]:
        return (None,), (self.w, self.b, self.activation,)

    @classmethod
    def tree_unflatten(
        cls, aux: Tuple[Tensor, Tensor, ActivationEnum], params: Tuple[None]
    ) -> "FrozenLinear":
        return cls.construct(w=aux[0], b=aux[1], activation=aux[2])


class Dropout(Layer):
    """Dropout Layer. If in train mode, keeps input activations at given probability rate,
    otherwise returns inputs directly"""

    rng: RNGWrapper
    keep: float = 0.5
    mode: Mode = Mode.train

    @jit
    def __call__(self, inputs: Tensor) -> Tensor:
        """If in train mode, keeps input activations at rate,
        otherwise returns directly"""

        if self.mode == Mode.eval:
            return inputs
        rng_key = self.rng.to_prng()
        mask = random.bernoulli(rng_key, self.keep, inputs.shape)
        return np.where(mask, inputs / self.keep, 0)

    def to_eval(self) -> "Dropout":
        return Dropout(rng=self.rng, keep=self.keep, mode=Mode.eval)

    def to_train(self) -> "Dropout":
        return Dropout(rng=self.rng.split(), keep=self.keep, mode=Mode.train)

    def tree_flatten(self) -> Tuple[Tuple[None], Tuple[RNGWrapper, float, Mode]]:
        return (None,), (self.rng, self.keep, self.mode)

    @classmethod
    def tree_unflatten(
        cls, aux: Tuple[RNGWrapper, float, Mode], params: Tuple[None]
    ) -> "Dropout":
        return cls.construct(rng=aux[0], keep=aux[1], mode=aux[2])


class Embedding(Layer):

    embedding_matrix: Tensor

    @jit
    def __call__(self, inputs: Tensor) -> Tensor:
        return self.embedding_matrix[inputs]

    @classmethod
    def initialize(
        cls,
        vocab_size: int,
        hidden_dim: int,
        key: Tensor,
        initializer: InitializerEnum = InitializerEnum.glorot_normal,
    ) -> "Embedding":
        vectors = INITIALIZERS[initializer](key, shape=(vocab_size, hidden_dim))
        vectors = ops.index_update(vectors, ops.index[0, :], 0.0)
        return cls(embedding_matrix=vectors)

    def tree_flatten(self) -> Tuple[Tuple[Tensor], None]:
        return (self.embedding_matrix,), None

    @classmethod
    def tree_unflatten(cls, aux: None, params: Tuple[Tensor]) -> "Embedding":
        return cls.construct(embedding_matrix=params[0])


class FrozenEmbedding(Embedding):
    """Untrainable Embedding Layer for pretrained embedding"""

    def tree_flatten(self) -> Tuple[Tuple[None], Tensor]:
        return (None,), self.embedding_matrix

    @classmethod
    def tree_unflatten(cls, aux: Tensor, params: Tuple[None]) -> "FrozenEmbedding":
        return cls.construct(embedding_matrix=aux)


class LSTMCell(Layer):

    Wf: Tensor
    bf: Tensor
    Wi: Tensor
    bi: Tensor
    Wc: Tensor
    bc: Tensor
    Wo: Tensor
    bo: Tensor

    @classmethod
    def initialize(cls, input_dim: int, hidden_dim: int, key: Tensor) -> "LSTMCell":

        Wf_key, Wi_key, Wc_key, Wo_key = random.split(key, num=4)

        concat_size = input_dim + hidden_dim

        Wf = INITIALIZERS["glorot_normal"](Wf_key, shape=(hidden_dim, concat_size))
        bf = np.zeros(shape=(hidden_dim,))

        Wi = INITIALIZERS["glorot_normal"](Wi_key, shape=(hidden_dim, concat_size))
        bi = np.zeros(shape=(hidden_dim,))

        Wc = INITIALIZERS["glorot_normal"](Wc_key, shape=(hidden_dim, concat_size))
        bc = np.zeros(shape=(hidden_dim,))

        Wo = INITIALIZERS["glorot_normal"](Wo_key, shape=(hidden_dim, concat_size))
        bo = np.zeros(shape=(hidden_dim,))

        return cls(Wf=Wf, bf=bf, Wi=Wi, bi=bi, Wc=Wc, bc=bc, Wo=Wo, bo=bo)

    @jit
    def __call__(
        self, state: Tuple[Tensor, Tensor], inputs: Tensor
    ) -> Tuple[Tuple[Tensor, Tensor], Tensor]:

        h_prev, c_prev = state
        concat_vec = np.hstack((inputs, h_prev))

        f = nn.sigmoid(np.dot(self.Wf, concat_vec) + self.bf)
        i = nn.sigmoid(np.dot(self.Wi, concat_vec) + self.bi)
        C_bar = np.tanh(np.dot(self.Wc, concat_vec) + self.bc)

        c = f * c_prev + i * C_bar
        o = nn.sigmoid(np.dot(self.Wo, concat_vec) + self.bo)
        h = o * np.tanh(c)

        # hidden state vector is copied as output
        return (h, c), h

    def tree_flatten(self) -> Tuple[Tuple[Tensor, ...], None]:
        return (
            (self.Wf, self.bf, self.Wi, self.bi, self.Wc, self.bc, self.Wo, self.bo),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux: None, params: Tuple[Tensor, ...]) -> "LSTMCell":

        return cls.construct(
            Wf=params[0],
            bf=params[1],
            Wi=params[2],
            bi=params[3],
            Wc=params[4],
            bc=params[5],
            Wo=params[6],
            bo=params[7],
        )


class FrozenLSTMCell(LSTMCell):
    def tree_flatten(self) -> Tuple[Tuple[Tensor, ...], None]:
        return (
            (self.Wf, self.bf, self.Wi, self.bi, self.Wc, self.bc, self.Wo, self.bo),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux: None, params: Tuple[Tensor, ...]) -> "LSTMCell":

        return cls.construct(
            Wf=params[0],
            bf=params[1],
            Wi=params[2],
            bi=params[3],
            Wc=params[4],
            bc=params[5],
            Wo=params[6],
            bo=params[7],
        )
