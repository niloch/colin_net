"""
The canonical example of a function that can't be
learned with a simple linear model is XOR
"""
import jax.numpy as np

import wandb
from colin_net.config import Experiment
from colin_net.layers import Linear
from colin_net.tensor import Tensor

# Create Input Data and True Labels
inputs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

targets = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])


config = {
    "experiment_name": "xor_runs",
    "model_config": {
        "output_dim": 2,
        "input_dim": 2,
        "hidden_dim": 5,
        "num_hidden": 4,
        "activation": "tanh",
        "dropout_keep": 0.5,
    },
    "random_seed": 42,
    "loss": "mean_squared_error",
    "regularization": None,
    "optimizer": "adam",
    "iterator_type": "batch_iterator",
    "learning_rate": 0.001,
    "batch_size": 4,
    "global_step": 5000,
    "log_every": 10,
}


wandb.init(project="colin_net_xor", config=config, sync_tensorboard=True)

experiment = Experiment(**config)


# define accuracy calculation
def accuracy(actual: Tensor, predicted: Tensor) -> float:
    return np.mean(np.argmax(actual, axis=1) == np.argmax(predicted, axis=1))


for update_state in experiment.train(inputs, targets, inputs, targets):
    if update_state.iteration % experiment.log_every == 0:
        model = update_state.model
        train_predicted = model.predict_proba(inputs)
        train_accuracy = float(accuracy(targets, train_predicted))
        model = model.to_eval()
        predicted = model.predict_proba(inputs)
        acc_metric = float(accuracy(targets, predicted))
        update_state.log_writer.add_scalar(
            "train_accuracy", acc_metric, update_state.iteration
        )
        update_state.log_writer.add_scalar(
            "train_accuracy", train_accuracy, update_state.iteration
        )
        # print(f"Accuracy: {acc_metric}")
        bar = update_state.bar
        bar.set_description(f"acc:{acc_metric}, loss:{update_state.loss}")
        for i, layer in enumerate(model.layers):
            if isinstance(layer, Linear):
                wandb.log({f"layer_{i}_w": wandb.Histogram(layer.w)})
                wandb.log({f"layer_{i}_b": wandb.Histogram(layer.b)})
        update_state.log_writer.flush()
        if acc_metric >= 0.99:
            print("Achieved Perfect Prediction!")
            break
        model = model.to_train()


final_model = update_state.model
final_model.save(update_state.log_writer.logdir + "/final_model.pkl", overwrite=True)

# Display Predictions
final_model = final_model.to_eval()
probabilties = final_model.predict_proba(inputs)
for gold, prob, pred in zip(targets, probabilties, np.argmax(probabilties, axis=1)):

    print(gold, prob, pred)

accuracy_score = float(accuracy(targets, probabilties))
print("Accuracy: ", accuracy_score)
