"""
A Model is just a collection of layers.
It behaves a lot like a layer itself.
"""
import pickle
from pathlib import Path
from typing import List, Tuple, Union

import jax.numpy as np
from jax import jit, lax, nn, random, vmap

from colin_net.base import RNGWrapper
from colin_net.layers import (ActivationEnum, Dropout, Embedding, InitializerEnum,
                              Layer, Linear, LSTMCell)
from colin_net.tensor import Tensor

suffix = ".pkl"


class Model(Layer, is_abstract=True):
    output_dim: int

    """Abstract Class for Model. Enforces subclasses to implement
    __call__, tree_flatten, tree_unflatten, save, load and registered as Pytree"""

    def __call__(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def to_eval(self) -> "Model":
        return self

    def to_train(self) -> "Model":
        return self

    def randomize(self) -> "Model":
        return self

    def save(self, path: Union[str, Path], overwrite: bool = False) -> None:
        path = Path(path)
        if path.suffix != suffix:
            path = path.with_suffix(suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if overwrite:
                path.unlink()
            else:
                raise RuntimeError(f"File {path} already exists.")
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Model":
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        if path.suffix != suffix:
            raise ValueError(f"Not a {suffix} file: {path}")
        with open(path, "rb") as file:
            data = pickle.load(file)
        return data

    @jit
    def predict_proba(self, inputs: Tensor) -> Tensor:
        if self.output_dim > 1:
            return nn.softmax(self.__call__(inputs))
        else:
            return nn.sigmoid(self.__call__(inputs))


class MLP(Model):
    """Class for feed forward nets like Multilayer Perceptrons."""

    layers: List[Layer]
    input_dim: int
    output_dim: int

    @jit
    def predict(self, single_input: Tensor) -> Tensor:
        """Predict for a single instance by iterating over all the layers"""

        for layer in self.layers:
            single_input = layer(single_input)
        return single_input

    @jit
    def __call__(self, batched_inputs: Tensor) -> Tensor:
        """Batched Predictions"""

        return vmap(self.predict)(batched_inputs)

    @classmethod
    def create_mlp(
        cls,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_hidden: int,
        key: Tensor,
        activation: ActivationEnum = ActivationEnum.tanh,
        dropout_keep: float = None,
        initializer: InitializerEnum = InitializerEnum.normal,
    ) -> "MLP":
        key, subkey = random.split(key)
        layers: List[Layer] = [
            Linear.initialize(
                input_dim=input_dim,
                output_dim=hidden_dim,
                key=subkey,
                activation=activation,
                initializer=initializer,
            ),
        ]
        if dropout_keep:
            key, subkey = random.split(key)
            rng = RNGWrapper.from_prng(subkey)

            layers.append(Dropout(rng=rng, keep=dropout_keep))

        for _ in range(num_hidden - 2):
            key, subkey = random.split(key)
            layers.append(
                Linear.initialize(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    key=subkey,
                    activation=activation,
                    initializer=initializer,
                )
            )
            if dropout_keep:
                key, subkey = random.split(key)
                rng = RNGWrapper.from_prng(subkey)
                layers.append(Dropout(rng=rng, keep=dropout_keep))

        key, subkey = random.split(key)
        layers.append(
            Linear.initialize(
                input_dim=hidden_dim,
                output_dim=output_dim,
                key=subkey,
                activation=ActivationEnum.identity,
                initializer=initializer,
            )
        )

        return cls(layers=layers, input_dim=input_dim, output_dim=output_dim)

    @jit
    def to_eval(self) -> "MLP":
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dropout):
                new_layer = layer.to_eval()
                self.layers[i] = new_layer
        return MLP(
            layers=self.layers, input_dim=self.input_dim, output_dim=self.output_dim
        )

    @jit
    def to_train(self) -> "MLP":
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dropout):
                new_layer = layer.to_train()
                self.layers[i] = new_layer
        return MLP(
            layers=self.layers, input_dim=self.input_dim, output_dim=self.output_dim
        )

    def tree_flatten(self) -> Tuple[List[Layer], Tuple[int, int]]:
        return self.layers, (self.input_dim, self.output_dim)

    @classmethod
    def tree_unflatten(cls, aux: Tuple[int, int], params: List[Layer]) -> "MLP":
        return cls(layers=params, input_dim=aux[0], output_dim=aux[1])


class LSTMClassifier(Model):

    embeddings: Embedding
    cell: LSTMCell
    output_layer: Linear
    h_prev: Tensor
    c_prev: Tensor
    output_dim: int

    @classmethod
    def initialize(
        cls, *, vocab_size: int, hidden_dim: int, output_dim: int, key: Tensor
    ) -> "LSTMClassifier":
        key, subkey = random.split(key)
        embedding = Embedding.initialize(vocab_size, hidden_dim, subkey)

        key, subkey = random.split(key)
        cell = LSTMCell.initialize(
            input_dim=hidden_dim, hidden_dim=hidden_dim, key=subkey,
        )

        key, subkey = random.split(key)
        output_layer = Linear.initialize(
            input_dim=hidden_dim,
            output_dim=output_dim,
            activation=ActivationEnum.identity,
            key=subkey,
        )

        h_prev = np.zeros(shape=(hidden_dim,))
        c_prev = np.zeros(shape=(hidden_dim,))

        return cls(
            embeddings=embedding,
            cell=cell,
            output_layer=output_layer,
            h_prev=h_prev,
            c_prev=c_prev,
            output_dim=output_dim,
        )

    @jit
    def predict(self, single_input: Tensor) -> Tensor:
        h_prev = self.h_prev
        c_prev = self.c_prev

        @jit
        def wrapped_cell(
            state: Tuple[Tensor, Tensor], inputs: Tensor
        ) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
            return self.cell(state, inputs)

        sentence_embedding = self.embeddings(single_input)
        state_tuple, output_sequence = lax.scan(
            wrapped_cell, init=(h_prev, c_prev), xs=sentence_embedding
        )
        # Semantics of lax.scan
        # outputs = []
        # for word_vector in sentence_embedding:
        #     (h_prev, c_prev), output = self.cell((h_prev, c_prev), word_vector)
        #     outputs.append(output)
        # return (h_new, c_new), outputs

        h_final = output_sequence[-1]
        output = self.output_layer(h_final)

        return output

    @jit
    def __call__(self, batched_inputs: Tensor) -> Tensor:
        return vmap(self.predict)(batched_inputs)

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[Embedding, LSTMCell, Linear, Tensor, Tensor], int]:
        return (
            (self.embeddings, self.cell, self.output_layer, self.h_prev, self.c_prev,),
            self.output_dim,
        )

    @classmethod
    def tree_unflatten(
        cls, aux: int, params: Tuple[Embedding, LSTMCell, Linear, Tensor, Tensor],
    ) -> "LSTMClassifier":
        return cls.construct(
            embeddings=params[0],
            cell=params[1],
            output_layer=params[2],
            h_prev=params[3],
            c_prev=params[4],
            output_dim=aux,
        )


__all__ = ["Model", "MLP", "LSTMClassifier"]
