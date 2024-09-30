"""Main landshark commands."""

# Copyright 2019 CSIRO (Data61)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from typing import NamedTuple, Optional

import click
import json
import numpy as np
import tensorflow_datasets as tfds

from landshark import __version__, errors
from landshark import kerasmodel
from landshark.kerasmodel import train_test as keras_train_test
from landshark.model import (
    QueryConfig,
    TrainingConfig,
    predict,
    setup_query,
    setup_oos_query,
    setup_training,
    train_test,
)
from landshark.saver import overwrite_model_dir
from landshark.scripts.logger import configure_logging
from landshark.tifwrite import write_geotiffs
from landshark.util import points_per_batch, score
from landshark.tfread import dataset_fn

log = logging.getLogger(__name__)


class CliArgs(NamedTuple):
    """Arguments passed from the base command."""

    gpu: bool
    batchMB: float
    keras: bool


@click.group()
@click.version_option(version=__version__)
@click.option("--gpu/--no-gpu", default=False, help="Have tensorflow use the GPU")
@click.option(
    "--batch-mb",
    type=float,
    default=10,
    help="Approximate size in megabytes of data read per " "worker per iteration",
)
@click.option(
    "-v",
    "--verbosity",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Level of logging",
)
@click.option("--keras-model", is_flag=True, help="Use a tf.keras.Model configuration.")
@click.pass_context
def cli(
    ctx: click.Context, gpu: bool, verbosity: str, batch_mb: float, keras_model: bool
) -> int:
    """Train a model and use it to make predictions."""
    ctx.obj = CliArgs(gpu=gpu, batchMB=batch_mb, keras=keras_model)
    configure_logging(verbosity)
    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("Disabling GPU use: CUDA_VISIBLE_DEVICES='-1'")
    return 0


@cli.command()
@click.option(
    "--data",
    type=click.Path(exists=True),
    required=True,
    help="The traintest or trainvalidation folder containing the data",
)
@click.option(
    "--trainvalidation",
    type=click.BOOL,
    required=False,
    default=False,
    help="Whether the data folder is a traintes or trainvalidation folder",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="The model configuration file",
)
@click.option(
    "--epochs",
    type=click.IntRange(min=1),
    default=1,
    help="Epochs between testing the model.",
)
@click.option(
    "--batchsize", type=click.IntRange(min=1), default=1000, help="Training batch size"
)
@click.option(
    "--test_batchsize",
    type=click.IntRange(min=1),
    default=1000,
    help="Testing batch size",
)
@click.option(
    "--iterations",
    type=click.IntRange(min=1),
    default=None,
    help="number of training/testing iterations.",
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    default=None,
    help="Optional directory containing model checkpoints.",
)
@click.pass_context
def train(
    ctx: click.Context,
    data: str,
    trainvalidation: bool,
    config: str,
    epochs: int,
    batchsize: int,
    test_batchsize: int,
    iterations: Optional[int],
    checkpoint: Optional[str],
) -> None:
    """Train a model specified by a config file."""
    log.info("Ignoring batch-mb option, using specified or default batchsize")
    catching_f = errors.catch_and_exit(train_entrypoint)
    catching_f(
        data,
        trainvalidation,
        config,
        ctx.obj.keras,
        epochs,
        batchsize,
        test_batchsize,
        iterations,
        ctx.obj.gpu,
        checkpoint,
    )


def train_entrypoint(
    data: str,
    trainvalidation: bool,
    config: str,
    keras: bool,
    epochs: int,
    batchsize: int,
    test_batchsize: int,
    iterations: Optional[int],
    gpu: bool,
    checkpoint_dir: Optional[str],
) -> None:
    """Entry point for training function."""
    train_test_fn = keras_train_test if keras else train_test
    metadata, training_records, testing_records, model_dir, cf = setup_training(
        config, data, trainvalidation
    )
    if checkpoint_dir:
        overwrite_model_dir(model_dir, checkpoint_dir)

    training_params = TrainingConfig(epochs, batchsize, test_batchsize, gpu)
    training_params.save(model_dir)  # save for later use in tfp probability model prediction
    train_test_fn(
        training_records,
        testing_records,
        metadata,
        model_dir,
        sys.modules[cf],
        training_params,
        iterations,
    )


@cli.command("predict")
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to the model file",
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    required=True,
    help="Path to the trained model checkpoint",
)
@click.option(
    "--data",
    type=click.Path(exists=True),
    required=True,
    help="Path to the query data directory",
)
@click.option(
    "--proba",
    type=click.BOOL,
    required=False,
    default=False,
    help="Whether it's a probabilistic model",
)
@click.option(
    "--pred_ensemble_size",
    type=click.IntRange(min=5),
    required=False,
    default=1000,
    help="Number of samples for the prediction ensemble",
)
@click.pass_context
def run_predict(ctx: click.Context, config: str, checkpoint: str, data: str, proba: bool,
                pred_ensemble_size: int) -> None:
    """Predict using a learned model."""
    catching_f = errors.catch_and_exit(predict_entrypoint)
    catching_f(config, ctx.obj.keras, checkpoint, data, ctx.obj.batchMB, ctx.obj.gpu, proba,
               pred_ensemble_size)


def predict_entrypoint(
    config: str, keras: bool, checkpoint: str, data: str, batchMB: float, gpu: bool, proba: bool,
    pred_ensemble_size: int
) -> None:
    """Entrypoint for predict function."""
    if keras:
        from functools import partial
        log.info(f"Using prediction ensemble size of {pred_ensemble_size}")
        training_config = TrainingConfig.load(checkpoint)
        if proba:
            predict_fn = partial(
                kerasmodel.predict_tfp,
                pred_ensemble_size=pred_ensemble_size,
                training_config=training_config
            )
        else:
            predict_fn = partial(
                kerasmodel.predict,
                pred_ensemble_size=pred_ensemble_size,
                training_config=training_config
            )
    else:
        predict_fn = predict

    train_metadata, feature_metadata, query_records, strip, nstrips, cf = setup_query(
        config, data, checkpoint
    )

    query_batchsize = points_per_batch(train_metadata.features, batchMB)

    params = QueryConfig(query_batchsize, gpu)
    y_dash_it = predict_fn(
        checkpoint, sys.modules[cf], train_metadata, query_records, params
    )

    write_geotiffs(
        y_dash_it,
        checkpoint,
        feature_metadata.image,
        tag="{}of{}".format(strip, nstrips),
    )


@cli.command("predict_oos")
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to the model file",
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    required=True,
    help="Path to the trained model checkpoint",
)
@click.option(
    "--data",
    type=click.Path(exists=True),
    required=True,
    help="Path to the hdf5 file containing the oos targets",
)
@click.option(
    "--proba",
    type=click.BOOL,
    required=False,
    default=False,
    help="Whether it's a probabilistic model",
)
@click.option(
    "--pred_ensemble_size",
    type=click.IntRange(min=3),
    required=False,
    default=1000,
    help="Number of samples for the prediction ensemble",
)
@click.pass_context
def run_oos_predict(ctx: click.Context, config: str, checkpoint: str, data: str, proba: bool,
                    pred_ensemble_size: int) -> None:
    """Predict using a learned model."""
    catching_f = errors.catch_and_exit(oos_predict_entrypoint)
    catching_f(config, ctx.obj.keras, checkpoint, data, ctx.obj.batchMB, ctx.obj.gpu, proba,
               pred_ensemble_size)


def oos_predict_entrypoint(
        config: str, keras: bool, checkpoint: str, data: str, batchMB: float, gpu: bool, proba: bool,
        pred_ensemble_size: int
) -> None:
    """Entrypoint for predict function."""
    if keras:
        from functools import partial
        log.info(f"Using prediction ensemble size of {pred_ensemble_size}")
        training_config = TrainingConfig.load(checkpoint)
        if proba:
            predict_fn = partial(
                kerasmodel.predict_tfp,
                pred_ensemble_size=pred_ensemble_size,
                training_config=training_config
            )
        else:
            predict_fn = partial(
                kerasmodel.predict,
                pred_ensemble_size=pred_ensemble_size,
                training_config=training_config
            )
    else:
        predict_fn = predict

    train_metadata, oos_metadata, oos_records, cf = setup_oos_query(config, data, checkpoint)

    query_batchsize = points_per_batch(train_metadata.features, batchMB)

    params = QueryConfig(query_batchsize, gpu)
    y_dash_it = predict_fn(
        checkpoint, sys.modules[cf], train_metadata, oos_records, params
    )
    y_pred = [y for y in y_dash_it]  # evaluation
    labels = [k for k, v in y_pred[0].items()]
    xtest = dataset_fn(
        oos_records, 1000, oos_metadata.features, oos_metadata.targets
    )()
    ds_numpy = tfds.as_numpy(xtest)

    y_true_numpy = np.vstack([ex[1] for ex in ds_numpy])

    y_pred_numpy = np.vstack([np.hstack(list(v.values())) for v in y_pred])
    scores = score(labels, y_true_numpy, y_pred_numpy)

    score_string = "OOS Validation complete:\n"
    for label, scrs in scores.items():
        score_string += "{}\n".format(label)
        for metric, sc in scrs.items():
            score_string += "{}\t= {}\n".format(metric, sc)
    log.info(score_string)

    score_path = os.path.join(checkpoint, "oos_validation_scores.json")
    with open(score_path, "w") as f:
        json.dump({k: str(v) for k, v in scores.items()}, f, indent=4)


if __name__ == "__main__":
    cli()
