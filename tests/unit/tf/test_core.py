from typing import List

import pytest
import tensorflow as tf
from tensorflow.keras import mixed_precision

import merlin.models.tf as ml
from merlin.io.dataset import Dataset
from merlin.models.tf.utils import testing_utils
from merlin.schema import Tags


def test_filter_features(tf_con_features):
    features = ["a", "b"]
    con = ml.Filter(features)(tf_con_features)

    assert list(con.keys()) == features


def test_as_tabular(tf_con_features):
    name = "tabular"
    con = ml.AsTabular(name)(tf_con_features)

    assert list(con.keys()) == [name]


def test_tabular_block(tf_con_features):
    _DummyTabular = ml.TabularBlock

    tabular = _DummyTabular()

    assert tabular(tf_con_features) == tf_con_features
    assert tabular(tf_con_features, aggregation="concat").shape[1] == 6
    assert tabular(tf_con_features, aggregation=ml.ConcatFeatures()).shape[1] == 6

    tabular_concat = _DummyTabular(aggregation="concat")
    assert tabular_concat(tf_con_features).shape[1] == 6

    tab_a = ["a"] >> _DummyTabular()
    tab_b = ["b"] >> _DummyTabular()

    assert tab_a(tf_con_features, merge_with=tab_b, aggregation="stack").shape[1] == 1


@pytest.mark.parametrize("pre", [None])
@pytest.mark.parametrize("post", [None])
@pytest.mark.parametrize("aggregation", [None, "concat"])
@pytest.mark.parametrize("include_schema", [True, False])
def test_serialization_continuous_features(
    testing_data: Dataset, pre, post, aggregation, include_schema
):
    schema = None
    if include_schema:
        schema = testing_data.schema

    inputs = ml.TabularBlock(pre=pre, post=post, aggregation=aggregation, schema=schema)

    copy_layer = testing_utils.assert_serialization(inputs)

    keep_cols = ["user_id", "item_id", "event_hour_sin", "event_hour_cos"]
    tf_tabular_data = ml.sample_batch(testing_data, batch_size=100, include_targets=False)
    for k in list(tf_tabular_data.keys()):
        if k not in keep_cols:
            del tf_tabular_data[k]

    assert copy_layer(tf_tabular_data) is not None
    assert inputs.pre.__class__.__name__ == copy_layer.pre.__class__.__name__
    assert inputs.post.__class__.__name__ == copy_layer.post.__class__.__name__
    assert inputs.aggregation.__class__.__name__ == copy_layer.aggregation.__class__.__name__
    if include_schema:
        assert inputs.schema == schema


class DummyFeaturesBlock(ml.Block):
    def add_features_to_context(self, feature_shapes) -> List[str]:
        return [Tags.ITEM_ID.value]

    def call(self, inputs, **kwargs):
        items = self.context[Tags.ITEM_ID]
        emb_table = self.context.get_embedding(Tags.ITEM_ID)
        item_embeddings = tf.gather(emb_table, tf.cast(items, tf.int32))
        if tf.rank(item_embeddings) == 3:
            item_embeddings = tf.squeeze(item_embeddings)

        return inputs * item_embeddings

    def compute_output_shape(self, input_shapes):
        return input_shapes

    @property
    def item_embedding_table(self):
        return self.context.get_embedding(Tags.ITEM_ID)


def test_block_context(ecommerce_data: Dataset):
    inputs = ml.InputBlock(ecommerce_data.schema)
    dummy = DummyFeaturesBlock()
    block = inputs.connect(ml.MLPBlock([64]), dummy, context=ml.ModelContext())
    out = block(ml.sample_batch(ecommerce_data, batch_size=100, include_targets=False))

    embeddings = inputs.select_by_name(Tags.CATEGORICAL.value)
    assert (
        dummy.context.get_embedding(Tags.ITEM_ID).shape
        == embeddings.embedding_tables[Tags.ITEM_ID.value].shape
    )

    assert out.shape[-1] == 64


@pytest.mark.parametrize("run_eagerly", [True])
def test_block_context_model(ecommerce_data: Dataset, run_eagerly: bool, tmp_path):
    dummy = DummyFeaturesBlock()
    model = ml.Model(
        ml.InputBlock(ecommerce_data.schema),
        ml.MLPBlock([64]),
        dummy,
        ml.BinaryClassificationTask("click"),
    )

    model.compile(optimizer="adam", run_eagerly=run_eagerly)
    model.fit(ecommerce_data, batch_size=50, epochs=1)
    model.save(str(tmp_path))

    copy_model = tf.keras.models.load_model(str(tmp_path))
    assert copy_model.context == copy_model.block.layers[0].context
    assert list(copy_model.context._feature_names) == ["item_id"]
    assert len(dict(copy_model.context._feature_dtypes)) == 23

    copy_model.compile(optimizer="adam", run_eagerly=run_eagerly)
    # TODO: Fix prediction-task output name so that we can retrain a model after saving
    # copy_model.fit(ecommerce_data.tf_dataloader(), epochs=1)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_block_context_model_fp16(ecommerce_data: Dataset, run_eagerly: bool, num_epochs=2):
    mixed_precision.set_global_policy("mixed_float16")
    model = ml.Model(
        ml.InputBlock(ecommerce_data.schema),
        ml.MLPBlock([32]),
        ml.BinaryClassificationTask("click"),
    )
    model.compile(optimizer="adam", run_eagerly=run_eagerly)
    mixed_precision.set_global_policy("float32")
    losses = model.fit(ecommerce_data, batch_size=100, epochs=2)
    assert len(losses.epoch) == num_epochs
    assert all(measure >= 0 for metric in losses.history for measure in losses.history[metric])