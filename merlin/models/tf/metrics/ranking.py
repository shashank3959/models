#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Adapted from source code: https://github.com/karlhigley/ranking-metrics-torch
from abc import abstractmethod
from typing import List, Sequence

import numpy as np
import tensorflow as tf

from . import metrics_registry

METRIC_PARAMETERS_DOCSTRING = """
    scores : tf.Tensor
        A tensor with shape (batch_size, n_items) corresponding to
        the ranking scores of items.
    labels : tf.Tensor
        A tensor with shape (batch_size, n_items) corresponding to
        the one-hot representation of true labels.
"""


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class RankingMetric(tf.keras.metrics.Metric):
    """
    Metric wrapper for computing ranking metrics@K for session-based task.
    Parameters
    ----------
    top_ks : list, default [2, 5])
        list of cutoffs
    """

    def __init__(
        self,
        top_ks: Sequence[int],
        name=None,
        dtype=None,
        **kwargs,
    ):
        super(RankingMetric, self).__init__(name=name, **kwargs)
        self.top_ks = top_ks
        # Store the mean vector of the batch metrics (for each cut-off at topk) in ListWrapper
        self.metric_mean: List[tf.Tensor] = []
        self.accumulator = tf.Variable(
            tf.zeros(shape=[1, len(self.top_ks)]),
            trainable=False,
            shape=tf.TensorShape([None, tf.compat.v1.Dimension(len(self.top_ks))]),
        )

    def get_config(self):
        config = {"top_ks": self.top_ks}
        base_config = super(RankingMetric, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _build(self, shape):
        bs = shape[0]
        variable_shape = [bs, tf.compat.v1.Dimension(len(self.top_ks))]
        self.accumulator.assign(tf.zeros(variable_shape))

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, **kwargs):
        # Computing the metrics at different cut-offs
        # init batch accumulator
        self._build(shape=tf.shape(y_pred))
        # TODO solve applying check_inputs in graph-mode
        # y_true, y_pred = check_inputs(y_pred, y_true)
        self._metric(
            scores=tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]]),
            labels=y_true,
        )
        self.metric_mean.append(self.accumulator)

    def result(self):
        # Computing the mean of the batch metrics (for each cut-off at topk)
        return tf.reduce_mean(tf.concat(self.metric_mean, axis=0), axis=0)

    def reset_state(self):
        self.metric_mean = []

    @abstractmethod
    def _metric(self, scores: tf.Tensor, labels: tf.Tensor, **kwargs):
        """
        Update `self.accumulator` with the ranking metric of
        prediction scores and one-hot labels for different cut-offs `ks`.
        This method should be overridden by subclasses.
        Parameters
        ----------
        {METRIC_PARAMETERS_DOCSTRING}
        """
        raise NotImplementedError

    def metric_fn(self, scores: tf.Tensor, labels: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Compute ranking metric over predictions and one-hot targets for different cut-offs.
        Parameters
        ----------
        {METRIC_PARAMETERS_DOCSTRING}
        """
        self._build(shape=tf.shape(scores))
        self._metric(scores=tf.reshape(scores, [-1, tf.shape(scores)[-1]]), labels=labels, **kwargs)
        return self.accumulator


@metrics_registry.register_with_multiple_names("precision_at", "precision")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class PrecisionAt(RankingMetric):
    def __init__(self, top_ks=None, **kwargs):
        super(PrecisionAt, self).__init__(top_ks=top_ks, **kwargs)

    def _metric(self, scores: tf.Tensor, labels: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Compute precision@K for each provided cutoff in ks
        Parameters
        ----------
        {METRIC_PARAMETERS_DOCSTRING}
        """
        ks = tf.convert_to_tensor(self.top_ks)
        bs = tf.shape(scores)[0]

        for index in range(int(tf.shape(ks)[0])):
            k = ks[index]
            rows_ids = tf.range(bs, dtype=tf.int64)
            indices = tf.concat(
                [
                    tf.expand_dims(rows_ids, 1),
                    tf.cast(index, tf.int64) * tf.ones([bs, 1], dtype=tf.int64),
                ],
                axis=1,
            )

            self.accumulator.scatter_nd_update(
                indices=indices, updates=tf.reduce_sum(labels[:, : int(k)], axis=1) / float(k)
            )


@metrics_registry.register_with_multiple_names("recall_at", "recall")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class RecallAt(RankingMetric):
    def __init__(self, top_ks: Sequence[int], **kwargs):
        super(RecallAt, self).__init__(top_ks=top_ks, **kwargs)

    def _metric(self, scores: tf.Tensor, labels: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Compute recall@K for each provided cutoff in ks
        Parameters
        ----------
        {METRIC_PARAMETERS_DOCSTRING}
        """
        ks = tf.convert_to_tensor(self.top_ks)

        # Compute recalls at K
        num_relevant = tf.reduce_sum(labels, axis=-1)
        rel_indices = tf.where(num_relevant != 0)
        rel_count = tf.gather_nd(num_relevant, rel_indices)

        if tf.shape(rel_indices)[0] > 0:
            for index in range(int(tf.shape(ks)[0])):
                k = ks[index]
                rel_labels = tf.cast(tf.gather_nd(labels, rel_indices)[:, : int(k)], tf.float32)
                batch_recall_k = tf.cast(
                    tf.reshape(
                        tf.math.divide(tf.reduce_sum(rel_labels, axis=-1), rel_count),
                        (len(rel_indices), 1),
                    ),
                    tf.float32,
                )
                # Ensuring type is double, because it can be float if --fp16

                update_indices = tf.concat(
                    [
                        rel_indices,
                        tf.expand_dims(
                            tf.cast(index, tf.int64) * tf.ones(tf.shape(rel_indices)[0], tf.int64),
                            -1,
                        ),
                    ],
                    axis=1,
                )
                self.accumulator.scatter_nd_update(
                    indices=update_indices, updates=tf.reshape(batch_recall_k, (-1,))
                )


@metrics_registry.register_with_multiple_names("avg_precision_at", "avg_precision", "map")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class AvgPrecisionAt(RankingMetric):
    def __init__(self, top_ks: Sequence[int], **kwargs):
        super(AvgPrecisionAt, self).__init__(top_ks=top_ks, **kwargs)
        max_k = tf.reduce_max(self.top_ks)
        self.precision_at = PrecisionAt(top_ks=1 + np.array((range(max_k)))).metric_fn

    def _metric(self, scores: tf.Tensor, labels: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Compute average precision @K for provided cutoff in ks
        Parameters
        ----------
        {METRIC_PARAMETERS_DOCSTRING}
        """
        ks = tf.convert_to_tensor(self.top_ks)

        num_relevant = tf.reduce_sum(labels, axis=-1)

        bs = tf.shape(scores)[0]
        precisions = self.precision_at(scores, labels)
        rel_precisions = precisions * labels

        for index in range(int(tf.shape(ks)[0])):
            k = ks[index]
            tf_total_prec = tf.reduce_sum(rel_precisions[:, :k], axis=1)
            clip_value = tf.clip_by_value(
                num_relevant, clip_value_min=1, clip_value_max=tf.cast(k, tf.float32)
            )

            rows_ids = tf.range(bs, dtype=tf.int64)
            indices = tf.concat(
                [
                    tf.expand_dims(rows_ids, 1),
                    tf.cast(index, tf.int64) * tf.ones([bs, 1], dtype=tf.int64),
                ],
                axis=1,
            )
            self.accumulator.scatter_nd_update(indices=indices, updates=tf_total_prec / clip_value)


@metrics_registry.register_with_multiple_names("dcg_at", "dcg")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class DCGAt(RankingMetric):
    def __init__(self, top_ks, **kwargs):
        super(DCGAt, self).__init__(top_ks=top_ks, **kwargs)

    def _metric(
        self, scores: tf.Tensor, labels: tf.Tensor, log_base: int = 2, **kwargs
    ) -> tf.Tensor:
        """
        Compute discounted cumulative gain @K for each provided cutoff in ks
        (ignoring ties)
        Parameters
        ----------
        {METRIC_PARAMETERS_DOCSTRING}
        """
        ks = tf.convert_to_tensor(self.top_ks)

        # Compute discounts
        max_k = tf.reduce_max(ks)
        discount_positions = tf.cast(tf.range(max_k), tf.float32)
        discount_log_base = tf.math.log(tf.convert_to_tensor([log_base], dtype=tf.float32))

        discounts = 1 / (tf.math.log(discount_positions + 2) / discount_log_base)
        bs = tf.shape(scores)[0]
        # Compute DCGs at K
        for index in range(len(self.top_ks)):
            k = ks[index]
            m = labels[:, :k] * tf.repeat(
                tf.expand_dims(discounts[:k], 0), tf.shape(labels)[0], axis=0
            )
            rows_ids = tf.range(bs, dtype=tf.int64)
            indices = tf.concat(
                [
                    tf.expand_dims(rows_ids, 1),
                    tf.cast(index, tf.int64) * tf.ones([bs, 1], dtype=tf.int64),
                ],
                axis=1,
            )

            self.accumulator.scatter_nd_update(
                indices=indices, updates=tf.cast(tf.reduce_sum(m, axis=1), tf.float32)
            )
            # Ensuring type is double, because it can be float if --fp16


@metrics_registry.register_with_multiple_names("ndcg_at", "ndcg")
@tf.keras.utils.register_keras_serializable(package="merlin.models")
class NDCGAt(RankingMetric):
    def __init__(self, top_ks: Sequence[int], **kwargs):
        super(NDCGAt, self).__init__(top_ks=top_ks, **kwargs)
        self.dcg_at = DCGAt(top_ks).metric_fn

    def _metric(
        self, scores: tf.Tensor, labels: tf.Tensor, log_base: int = 2, **kwargs
    ) -> tf.Tensor:
        """
        Compute normalized discounted cumulative gain @K for each provided cutoffs in ks
        (ignoring ties)
        Parameters
        ----------
        {METRIC_PARAMETERS_DOCSTRING}
        """

        # Compute discounted cumulative gains
        gains = self.dcg_at(labels=labels, scores=scores, log_base=log_base)
        self.accumulator.assign(gains)
        normalizing_gains = self.dcg_at(labels=labels, scores=labels, log_base=log_base)

        # Prevent divisions by zero
        relevant_pos = tf.where(normalizing_gains != 0)
        tf.where(normalizing_gains == 0, 0.0, gains)

        updates = tf.gather_nd(self.accumulator, relevant_pos) / tf.gather_nd(
            normalizing_gains, relevant_pos
        )
        self.accumulator.scatter_nd_update(relevant_pos, updates)


def check_inputs(scores, labels):
    if tf.rank(scores) != 2:
        raise ValueError(f"scores must be 2-D tensor, (got {scores.shape})")

    if tf.rank(labels) != 2:
        raise ValueError(f"labels must be 2-D tensor, (got {labels.shape})")

    scores.get_shape().assert_is_compatible_with(labels.get_shape())

    return (tf.cast(scores, tf.float32), tf.cast(labels, tf.float32))


def process_metrics(metrics, prefix=""):
    metrics_proc = {}
    for metric in metrics:
        results = metric.result()
        if getattr(metric, "top_ks", None):
            for i, ks in enumerate(metric.top_ks):
                metrics_proc.update(
                    {f"{prefix}{metric.name.split('_')[0]}@{ks}": tf.gather(results, i)}
                )
        else:
            metrics_proc[metric.name] = results

    return metrics_proc


def ranking_metrics(top_ks: Sequence[int], **kwargs) -> Sequence[RankingMetric]:
    return NDCGAt(top_ks, **kwargs), RecallAt(top_ks, **kwargs), AvgPrecisionAt(top_ks, **kwargs)