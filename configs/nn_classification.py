"""Generic classification config file."""

from typing import Dict, Optional, cast

import tensorflow as tf
from tensorflow.estimator import ModeKeys

from landshark import config as utils
from landshark.metadata import CategoricalTarget, Training


def model(mode: ModeKeys,
          X_con: Optional[Dict[str, tf.Tensor]],
          X_con_mask: Optional[Dict[str, tf.Tensor]],
          X_cat: Optional[Dict[str, tf.Tensor]],
          X_cat_mask: Optional[Dict[str, tf.Tensor]],
          Y: tf.Tensor,
          image_indices: tf.Tensor,
          coordinates: tf.Tensor,
          metadata: Training,
          ) -> tf.estimator.EstimatorSpec:
    """
    Describe the specification of a Tensorflow custom estimator model.

    This function must be implemented in all configurations. It is almost
    exactly the model function passed to a custom Tensorflow estimator,
    apart from having more convenient input arguments.
    See https://www.tensorflow.org/guide/custom_estimators

    Parameters
    ----------
    mode : tf.estimator.ModeKeys
        One of TRAIN, TEST or EVAL, describing in which context this code
        is being run by the estimator.
    coordinates : tf.Tensor
        A (?, 2) the world coordinates (x, y) of features
    X_con : dict
        A dictionary of tensors containing the continuous feature columns.
        Indexed by column names.
    X_con_mask : dict
        A dictionary of tensors containing the masks for the continuous
        feature columns.
    X_cat : dict
        A dictionary of tensors containing the categorical feature columns.
        Indexed by column names.
    X_cat_mask : dict
        A dictionary of tensors containing the masks for the categorical
        feature columns.
    Y : tf.Tensor
        A (?, k) tensor giving the k targets for the prediction.
    image_indices : tf.Tensor
        A (?, 2) the image coordinates of features
    coordinates : tf.Tensor
        A (?, 2) the world coordinates (x, y) of features
    metadata : Metadata
        The metadata object has comprehensive information about the features
        and targets useful for model building (for example, the number of
        possible values for each categorical column). For more details
        check the landshark documentation.

    Returns
    -------
    tf.estimator.EstimatorSpec
        An EstimatorSpec object describing the model. For details check
        the Tensorflow custom estimator howto.

    """
    # Single-task classification
    nvalues_target = cast(CategoricalTarget, metadata.targets).nvalues[0]

    inputs_list = []
    if X_con:
        assert X_con_mask
        # let's 0-impute continuous columns
        X_con = {k: utils.value_impute(X_con[k], X_con_mask[k],
                                       tf.constant(0.0)) for k in X_con}

        # just concatenate the patch pixels as more features
        X_con = {k: utils.flatten_patch(v) for k, v in X_con.items()}

        # convenience function for catting all columns into tensor
        inputs_con = utils.continuous_input(X_con)
        inputs_list.append(inputs_con)

    if X_cat:
        assert X_cat_mask and metadata.features.categorical
        # impute as an extra category
        extra_cat = {
            k: metadata.features.categorical.columns[k].mapping.shape[0]
            for k in X_cat
        }
        X_cat = {k: utils.value_impute(
            X_cat[k], X_cat_mask[k], tf.constant(extra_cat[k]))
            for k in X_cat}
        X_cat = {k: utils.flatten_patch(v) for k, v in X_cat.items()}

        nvalues = {k: v.nvalues + 1
                   for k, v in metadata.features.categorical.columns.items()}
        embedding_dims = {k: 3 for k in X_cat.keys()}
        inputs_cat = utils.categorical_embedded_input(X_cat, nvalues,
                                                      embedding_dims)
        inputs_list.append(inputs_cat)

    # Build a simple 2-layer network
    inputs = tf.concat(inputs_list, axis=1)
    l1 = tf.layers.dense(inputs, units=64, activation=tf.nn.relu)
    l2 = tf.layers.dense(l1, units=32, activation=tf.nn.relu)

    # Get some predictions for the labels
    phi = tf.layers.dense(l2, units=nvalues_target,
                          activation=None)

    # geottiff doesn't support 64bit output from argmax
    predicted_classes = tf.cast(tf.argmax(phi, 1), tf.uint8)

    # Compute predictions.
    if mode == ModeKeys.PREDICT:
        predictions = {"predictions_{}".format(
            metadata.targets.labels[0]): predicted_classes}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Use a loss for training
    Y = Y[:, 0]
    ll_f = tf.distributions.Categorical(logits=phi)
    loss = -1 * tf.reduce_mean(ll_f.log_prob(Y))
    tf.summary.scalar("loss", loss)

    # Compute evaluation metrics.
    acc = tf.metrics.accuracy(labels=Y, predictions=predicted_classes)
    metrics = {"accuracy": acc}

    if mode == ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss,
                                          eval_metric_ops=metrics)

    # For training, use Adam to learn
    assert mode == ModeKeys.TRAIN
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)