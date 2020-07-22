import jax.numpy as np
import pytest
from jax import random, vmap
from numpy import testing

from colin_net.base import RNGWrapper, Tensor
from colin_net.layers import Dropout, Embedding, Linear
from colin_net.nn import MLP


@pytest.fixture
def single_input() -> Tensor:
    return np.ones(shape=(2,))


@pytest.fixture
def batched_inputs(single_input: Tensor) -> Tensor:
    return np.tile(single_input, (4, 1))


@pytest.fixture
def rng() -> RNGWrapper:
    return RNGWrapper.from_prng(random.PRNGKey(42))


@pytest.fixture
def dropout(rng: RNGWrapper) -> Dropout:
    return Dropout(keep=0.5, mode="train", rng=rng)


def test_dropout(single_input: Tensor, dropout: Dropout) -> None:

    outputs = dropout(single_input)

    testing.assert_array_equal(outputs, np.array([2.0, 0.0]))

    dropout = dropout.to_eval()

    outputs = dropout(single_input)

    testing.assert_array_equal(outputs, np.array([1.0, 1.0]))

    dropout = dropout.to_train()

    outputs = dropout(single_input)

    testing.assert_array_equal(outputs, np.array([0.0, 2.0]))


def test_mlp_dropout(batched_inputs: Tensor, dropout: Dropout) -> None:

    simplenn = MLP(layers=[dropout], input_dim=2, output_dim=2)

    outputs = simplenn(batched_inputs)

    testing.assert_array_equal(
        outputs, np.array([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    )

    simplenn = simplenn.to_eval()
    outputs = simplenn(batched_inputs)

    testing.assert_array_equal(
        outputs, np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    )

    simplenn = simplenn.to_train()
    outputs = simplenn(batched_inputs)

    testing.assert_array_equal(
        outputs, np.array([[0.0, 2.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]])
    )


@pytest.fixture
def embedding(rng: RNGWrapper) -> Embedding:
    return Embedding(
        embedding_matrix=np.array(
            [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]]
        )
    )


def test_embedding(embedding: Embedding) -> None:
    indices = np.array([0, 0, 0, 1, 1, 1])

    outputs = embedding(indices)

    testing.assert_array_equal(
        outputs,
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0],
            ]
        ),
    )
