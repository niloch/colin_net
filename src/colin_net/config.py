import json
from abc import abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Sequence, Type, Union

import jax.numpy as np
from jax import random
from jax.interpreters.xla import DeviceArray
from jax.tree_util import tree_flatten
from pydantic import BaseModel

import wandb
from colin_net.data import ITERATORS, DataIterator, IteratorEnum
from colin_net.layers import ActivationEnum, InitializerEnum
from colin_net.loss import (
    LOSS_FUNCTIONS,
    REGULARIZATIONS,
    Loss,
    LossEnum,
    RegularizationEnum,
)
from colin_net.metrics import accuracy
from colin_net.nn import MLP, LSTMClassifier, Model
from colin_net.optim import OPTIMIZERS, Optimizer, OptimizerEnum
from colin_net.tensor import Tensor


def get_keys(d: Dict[str, Any]) -> List[str]:
    keys = []
    for k, v in d.items():
        if isinstance(v, DeviceArray):
            keys.append(k)
        elif isinstance(v, dict):
            keys.extend(get_keys(v))
        elif isinstance(v, list):
            keys.extend([key for ls in v for key in get_keys(ls)])
    return keys


def log_wandb(d: Dict[str, Any], step: int) -> None:
    try:
        wandb.log(d, step=step)
    except:
        pass


def save_wandb(filename: str) -> None:
    try:
        wandb.save(filename)
    except:
        pass


class ModelConfig(BaseModel):
    output_dim: int

    @abstractmethod
    def initialize(self, key: Tensor) -> Model:
        raise NotImplementedError


class MLPConfig(ModelConfig):
    input_dim: int
    hidden_dim: int
    num_hidden: int
    activation: ActivationEnum = ActivationEnum.tanh
    dropout_keep: Optional[float] = None
    initializer: InitializerEnum = InitializerEnum.normal

    def initialize(self, key: Tensor) -> MLP:
        return MLP.create_mlp(key=key, **self.dict())


class LSTMConfig(ModelConfig):
    hidden_dim: int
    vocab_size: int

    def initialize(self, key: Tensor) -> LSTMClassifier:
        return LSTMClassifier.initialize(key=key, **self.dict())


class UpdateState(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    step: int
    loss: float
    model: Model


class Experiment(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    experiment_name: str
    model_config: Union[MLPConfig, LSTMConfig]
    random_seed: int = 42
    loss: LossEnum = LossEnum.mean_squared_error
    regularization: Optional[RegularizationEnum] = None
    optimizer: OptimizerEnum = OptimizerEnum.sgd
    learning_rate: float = 0.01
    batch_size: int = 32
    global_step: int = 5000
    log_every: float = 100

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Experiment":
        return cls(**d)

    @classmethod
    def from_file(cls, filename: str) -> "Experiment":
        with open(filename) as infile:
            d = json.load(infile)
            return cls.from_dict(d)

    def get_rng_keys(self, num: int = 1) -> Tensor:
        # breakpoint()
        if getattr(self, "key", None) is None:
            self.key: Tensor = random.PRNGKey(self.random_seed)
        keys = random.split(self.key, num=num + 1)
        if num > 1:
            subkeys = keys[1:]
        else:
            subkeys = keys[1]
        self.key = keys[0]
        return subkeys

    def create_model(self) -> Model:
        subkey = self.get_rng_keys()
        return self.model_config.initialize(subkey)

    def create_loss_func(self) -> Loss:
        if self.regularization:
            loss = REGULARIZATIONS[self.regularization](LOSS_FUNCTIONS[self.loss])
        else:
            loss = LOSS_FUNCTIONS[self.loss]
        return loss

    def create_optimizer(self, model: Model, loss: Loss) -> Optimizer:
        optimizer_class: Type[Optimizer] = OPTIMIZERS[self.optimizer]
        return optimizer_class.initialize(model, loss, self.learning_rate)

    def create_iterator(
        self, iterator_type: str, inputs: Any, targets: Any, batch_size: int
    ) -> DataIterator:
        subkey = self.get_rng_keys()
        return ITERATORS[iterator_type](
            inputs=inputs, targets=targets, key=subkey, batch_size=self.batch_size
        )

    def train(
        self,
        train_X: Sequence[Any],
        train_Y: Sequence[Any],
        test_X: Sequence[Any],
        test_Y: Sequence[Any],
        iterator_type: str = IteratorEnum.batch_iterator,
    ) -> Iterator[UpdateState]:

        model = self.create_model()

        loss = self.create_loss_func()

        optimizer = self.create_optimizer(model, loss)

        # Instantiate the data Iterators
        train_iterator = self.create_iterator(
            iterator_type, train_X, train_Y, self.batch_size
        )
        test_iterator = self.create_iterator(
            iterator_type, test_X, test_Y, self.batch_size
        )

        step = 1
        train_loss_accumulator = 0.0

        prev_accuracy = 0
        while step < self.global_step:
            for batch in train_iterator:
                model = model.to_train()
                batch_loss, model = optimizer.step(batch.inputs, batch.targets)
                grads = optimizer.grads
                params, _ = tree_flatten(grads)
                layer_names = get_keys(model.dict())
                if len(layer_names) != len(set(layer_names)):
                    for i, (weights, name) in enumerate(zip(params, layer_names)):
                        log_wandb(
                            {f"grad_{name}_{i}": wandb.Histogram(weights)}, step=step
                        )
                else:
                    for weights, name in zip(params, layer_names):
                        log_wandb({f"grad_{name}": wandb.Histogram(weights)}, step=step)

                train_loss_accumulator += batch_loss
                if step % self.log_every == 0:
                    log_wandb(
                        {"train_loss": float(train_loss_accumulator / self.log_every)},
                        step=step,
                    )
                    train_loss_accumulator = 0.0

                    model = model.to_eval()
                    test_loss_accumulator = 0.0
                    test_pred_accumulator = []
                    for test_batch in test_iterator:
                        test_loss = loss(model, test_batch.inputs, test_batch.targets)
                        test_loss_accumulator += test_loss
                        test_pred_accumulator.append(
                            model.predict_proba(test_batch.inputs)
                        )

                    test_probs = np.vstack(test_pred_accumulator)
                    test_accuracy = accuracy(test_iterator.targets, test_probs)
                    if test_accuracy > prev_accuracy:
                        prev_accuracy = test_accuracy
                        model.save(
                            f"{self.experiment_name}_{step}_model.pkl", overwrite=True
                        )

                        save_wandb(f"{self.experiment_name}_{step}_model.pkl")
                    log_wandb(
                        {
                            "test_loss": float(test_loss_accumulator)
                            / test_iterator.num_batches
                        },
                        step=step,
                    )
                    log_wandb({"test_accuracy": float(test_accuracy)}, step=step)
                yield UpdateState(step=step, loss=batch_loss, model=model)
                step += 1
