"""Tests for alpacka.networks.keras."""

import functools
import os
import tempfile

import numpy as np
import pytest
from tensorflow import keras

from alpacka import data
from alpacka.networks import keras as keras_networks


@pytest.fixture
def keras_mlp():
    return keras_networks.KerasNetwork(
        network_signature=data.NetworkSignature(
            input=data.TensorSignature(shape=(13,)),
            output=data.TensorSignature(shape=(1,)),
        )
    )


@pytest.fixture
def multi_output_sig():
    return data.NetworkSignature(
        input=data.TensorSignature(shape=(4,)),
        output=((data.TensorSignature(shape=(3,)),) * 2),
    )


@pytest.fixture
def multi_output_mlp(multi_output_sig):
    return keras_networks.KerasNetwork(
        network_signature=multi_output_sig,
        model_fn=functools.partial(
            keras_networks.mlp, output_activation=(None, None)
        ),
    )


@pytest.fixture
def dataset():
    ((x_train, y_train), _) = keras.datasets.boston_housing.load_data()
    return (x_train, y_train)


@pytest.mark.parametrize('model_fn,input_shape,output_shape', [
    (keras_networks.mlp, (15,), (1,)),
    (keras_networks.convnet_mnist, (3, 3, 6), (1,)),
])
def test_model_valid(model_fn, input_shape, output_shape):
    network = keras_networks.KerasNetwork(
        model_fn=model_fn,
        network_signature=data.NetworkSignature(
            input=data.TensorSignature(shape=input_shape),
            output=data.TensorSignature(shape=output_shape),
        ),
    )
    batch_size = 7
    inp = np.zeros((batch_size,) + input_shape)
    out = network.predict(inp)
    assert out.shape == (batch_size,) + output_shape


def test_keras_mlp_train_epoch_on_boston_housing(keras_mlp, dataset):
    # Set up
    (x_train, y_train) = dataset
    x_train = x_train[:16]
    y_train = np.expand_dims(y_train[:16], 1)

    def data_stream():
        for _ in range(3):
            yield (x_train, y_train, np.ones_like(y_train))

    # Run
    metrics = keras_mlp.train(data_stream, 3)

    # Test
    assert 'loss' in metrics


def test_keras_mlp_predict_batch_on_boston_housing(keras_mlp, dataset):
    # Set up
    (data, _) = dataset
    data_batch = data[:16]

    # Run
    pred_batch = keras_mlp.predict(data_batch)

    # Test
    assert pred_batch.shape == (16, 1)


def test_keras_mlp_train_multi_output(multi_output_mlp, multi_output_sig):
    input_shape = (8,) + multi_output_sig.input.shape
    output_shape = (8,) + multi_output_sig.output[0].shape

    def data_stream():
        for _ in range(3):
            inp = np.zeros(input_shape)
            target = np.zeros(output_shape)
            mask = np.ones_like(target)
            yield (inp, (target, target), (mask, mask))

    # Just check that nothing errors out.
    multi_output_mlp.train(data_stream, 3)


def test_keras_mlp_predict_multi_output(multi_output_mlp, multi_output_sig):
    input_shape = (8,) + multi_output_sig.input.shape
    output_shape = (8,) + multi_output_sig.output[0].shape

    (out1, out2) = multi_output_mlp.predict(np.zeros(input_shape))
    assert out1.shape == output_shape
    assert out2.shape == output_shape


def test_keras_mlp_modify_weights(keras_mlp):
    # Set up
    new_params = keras_mlp.params
    for p in new_params:
        p *= 2

    # Run
    keras_mlp.params = new_params

    # Test
    for new, mlp in zip(new_params, keras_mlp.params):
        assert np.all(new == mlp)


def test_keras_mlp_save_weights(keras_mlp):
    # Set up, Run and Test
    with tempfile.NamedTemporaryFile() as temp_file:
        assert os.path.getsize(temp_file.name) == 0
        keras_mlp.save(temp_file.name)
        assert os.path.getsize(temp_file.name) > 0


def test_keras_mlp_restore_weights(keras_mlp):
    with tempfile.NamedTemporaryFile() as temp_file:
        # Set up
        orig_params = keras_mlp.params
        keras_mlp.save(temp_file.name)

        new_params = keras_mlp.params
        for p in new_params:
            p *= 2
        keras_mlp.params = new_params

        # Run
        keras_mlp.restore(temp_file.name)

        # Test
        for orig, mlp in zip(orig_params, keras_mlp.params):
            assert np.all(orig == mlp)


def test_clone_has_the_same_weights(keras_mlp):
    clone = keras_mlp.clone()
    np.testing.assert_equal(keras_mlp.params, clone.params)


def test_clone_has_independent_weights(keras_mlp):
    clone = keras_mlp.clone()
    new_params = keras_mlp.params
    for p in new_params:
        p += 1
    keras_mlp.params = new_params

    # Sadly, numpy doesn't have assert_not_equal.
    np.testing.assert_raises(
        AssertionError, np.testing.assert_equal, keras_mlp.params, clone.params
    )
