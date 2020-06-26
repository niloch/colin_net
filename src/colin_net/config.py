import json
from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, Iterator, Optional, Sequence, Union

import jax.numpy as np
from jax import random
from pydantic import BaseModel
from tensorboardX import SummaryWriter
from tqdm.autonotebook import tqdm

from colin_net.data import ITERATORS, IteratorEnum
from colin_net.layers import ActivationEnum, InitializerEnum
from colin_net.loss import LOSS_FUNCTIONS, REGULARIZATIONS, LossEnum, RegularizationEnum
from colin_net.metrics import accuracy
from colin_net.nn import MLP, LSTMClassifier, NeuralNet
from colin_net.optim import OPTIMIZERS, OptimizerEnum
from colin_net.tensor import Tensor

# from colin_net.train import Trainer


class NetConfig(BaseModel):
    output_dim: int

    @abstractmethod
    def initialize(self, key: Tensor) -> NeuralNet:
        raise NotImplementedError


class MLPConfig(NetConfig):
    input_dim: int
    hidden_dim: int
    num_hidden: int
    activation: ActivationEnum = ActivationEnum.tanh
    dropout_keep: Optional[float] = None
    initializer: InitializerEnum = InitializerEnum.normal

    def initialize(self, key: Tensor) -> MLP:
        return MLP.create_mlp(key=key, **self.dict())


class LSTMConfig(NetConfig):
    hidden_dim: int
    vocab_size: int

    def initialize(self, key: Tensor) -> LSTMClassifier:
        return LSTMClassifier.initialize(key=key, **self.dict())


class UpdateState(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    iteration: int
    loss: float
    net: NeuralNet
    train_writer: SummaryWriter
    test_writer: SummaryWriter


class Experiment(BaseModel):
    experiment_name: str
    net_config: Union[MLPConfig, LSTMConfig]
    random_seed: int = 42
    loss: LossEnum = LossEnum.mean_squared_error
    regularization: Optional[RegularizationEnum] = None
    optimizer: OptimizerEnum = OptimizerEnum.sgd
    iterator_type: IteratorEnum = IteratorEnum.batch_iterator
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

    def hparams(self) -> Dict[str, str]:

        return {
            key: str(val) for key, val in self.dict().items() if key != "net_config"
        }

    def train(
        self,
        train_X: Sequence[Any],
        train_Y: Sequence[Any],
        test_X: Sequence[Any],
        test_Y: Sequence[Any],
    ) -> Iterator[UpdateState]:

        # Intantiate the Net
        key = random.PRNGKey(self.random_seed)
        key, subkey = random.split(key)
        net = self.net_config.initialize(subkey)

        # Instantiate the Optimizer
        if self.regularization:
            loss = REGULARIZATIONS[self.regularization](LOSS_FUNCTIONS[self.loss])
        else:
            loss = LOSS_FUNCTIONS[self.loss]
        optimizer = OPTIMIZERS[self.optimizer].initialize(net, loss, self.learning_rate)

        # Instantiate the data Iterators
        key, subkey = random.split(key)
        train_iterator = ITERATORS[self.iterator_type](
            inputs=train_X, targets=train_Y, key=key, batch_size=self.batch_size
        )
        key, subkey = random.split(key)
        test_iterator = ITERATORS[self.iterator_type](
            inputs=test_X, targets=test_Y, key=key, batch_size=self.batch_size
        )

        # Create the loggers
        now = datetime.now().isoformat()
        train_writer = SummaryWriter(f"{self.experiment_name}/{now}-train")
        test_writer = SummaryWriter(f"{self.experiment_name}/{now}-test")
        train_writer.add_hparams(self.hparams(), {}, name="hparams")
        train_writer.add_text("Net", f"```{net.json()}```")

        updates = 0
        train_loss_accumulator = 0.0

        bar = tqdm(total=self.global_step)
        while updates < self.global_step:
            for batch in train_iterator:
                net = net.to_train()
                batch_loss, net = optimizer.step(batch.inputs, batch.targets)
                train_loss_accumulator += batch_loss
                if updates % self.log_every == 0:
                    net.save(f"{self.experiment_name}/{now}_{updates}_model.pkl")
                    train_writer.add_scalar(
                        "loss", float(train_loss_accumulator), updates
                    )
                    train_loss_accumulator = 0.0

                    net = net.to_eval()
                    test_loss_accumulator = 0.0
                    test_pred_accumulator = []
                    for test_batch in test_iterator:
                        test_loss = loss(net, test_batch.inputs, test_batch.targets)
                        test_loss_accumulator += test_loss
                        test_pred_accumulator.append(
                            net.predict_proba(test_batch.inputs)
                        )

                    test_probs = np.vstack(test_pred_accumulator)
                    test_accuracy = accuracy(test_iterator.targets, test_probs)
                    test_writer.add_scalar(
                        "loss", float(test_loss_accumulator), updates
                    )
                    print(test_accuracy)
                    test_writer.add_scalar("accuracy", float(test_accuracy), updates)
                bar.update()
                yield UpdateState(
                    iteration=updates,
                    loss=batch_loss,
                    net=net,
                    train_writer=train_writer,
                    test_writer=test_writer,
                )
                updates += 1
