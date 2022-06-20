"""Network interface implementation using the Keras framework."""

import functools

import gin
import numpy as np
import tensorflow as tf
from tensorflow import keras

from alpacka import data
from alpacka.networks import core


def _is_pointwise_loss(name):
    """Checks if a Keras loss name corresponds to a pointwise loss."""
    return name.startswith('mean_') or name in {
        'binary_crossentropy', 'hinge', 'squared_hinge', 'poisson', 'log_cosh',
        'huber_loss', 'cosine_similarity',
        'mse', 'mae', 'mape', 'msle', 'bce', 'logcosh', 'huber'
    }


def pointwise_loss(loss_fn):
    r"""Decorator marking a loss function as pointwise.

    A loss function is said to be pointwise, if it can be expressed as a sum of
    independent terms over the depth dimension.

    For instance, the mean squared error is pointwise:

        MSE(x, y) = \sum_i (x_i - y_i) ^ 2.

    The categorical cross-entropy given logits is not pointwise, because
    softmax(x)_i depends on all elements of x:

        CE(x, y) = \sum_i y_i * \log(\softmax(x)_i),

        where \softmax(x)_i = \exp(x_i) / (\sum_j \exp(x_j)).

    Alpacka allows specifying masks for training examples, that are of the same
    shape as the target. This is useful when we want to mask parts of the last
    dimension, e.g. when training Q-networks on targets provided for only one
    action at a time. Keras does not support this natively, as its loss
    functions sum over the last dimension.

    This decorator wraps Keras loss functions, adding a singleton dimension at
    the end of the tensors before passing them to the actual loss. The Keras
    loss will just squeeze this singleton dimension.

    Args:
        loss_fn: Function (y_true, y_pred) -> loss, where y_true, y_pred are
            tensors of shape (..., depth) and loss is a tensor of shape (...).

    Returns:
        Function (y_true, y_pred) -> loss, where all variables are tensors of
        the same shape.
    """
    def new_loss_fn(y_true, y_pred):
        assert y_true.shape.is_compatible_with(y_pred.shape)
        loss = loss_fn(y_true[..., None], y_pred[..., None])
        assert loss.shape.is_compatible_with(y_true.shape)
        return loss

    new_loss_fn.__name__ = loss_fn.__name__ + '_pointwise'
    return new_loss_fn


def interdependent_loss(loss_fn):
    """Decorator marking a loss function as interdependent.

    Any loss function that is not pointwise is interdependent - see the
    docstring for pointwise_loss.

    For interdependent losses, providing masks that differ over the last
    dimension does not make sense, as the loss terms depend on other terms
    anyway. But we still need to accept masks of the same shape as the targets,
    to have a consistent interface with pointwise losses. We just ignore the
    differences in the mask over the last dimension.

    This decorator wraps Keras loss functions, re-adding the last dimension
    to the output, and broadcasting it to match the shape of the mask.

    Args:
        loss_fn: Function (y_true, y_pred) -> loss, where y_true, y_pred are
            tensors of shape (..., depth) and loss is a tensor of shape (...).

    Returns:
        Function (y_true, y_pred) -> loss, where all variables are tensors of
        the same shape.
    """
    def new_loss_fn(y_true, y_pred):
        assert y_true.shape.is_compatible_with(y_pred.shape)
        loss = loss_fn(y_true, y_pred)
        return tf.broadcast_to(loss[..., None], shape=tf.shape(y_pred))

    new_loss_fn.__name__ = loss_fn.__name__ + '_interdependent'
    return new_loss_fn


def _wrap_loss(loss_or_name):
    if isinstance(loss_or_name, str):
        # Losses specified by name are assumed to be Keras losses, that need
        # wrapping.
        name = loss_or_name
        loss = keras.losses.get(name)
    elif 'tensorflow.python.keras' in loss_or_name.__module__:
        # Things defined in Keras are assumed to be Keras losses, that need
        # wrapping.
        loss = loss_or_name
        name = loss.name
    else:
        # All other losses are assumed to be custom, matching the interface
        # expected by Alpacka - see the docstring for pointwise_loss.
        return loss_or_name

    if _is_pointwise_loss(name):
        return pointwise_loss(loss)
    else:
        return interdependent_loss(loss)


class AddMask(keras.layers.Layer):
    """Creates a Keras mask from an (input, mask) pair."""

    supports_masking = True

    def compute_mask(self, inputs, mask=None):
        assert mask is None
        (_, mask) = inputs
        return mask

    def call(self, inputs, **kwargs):
        del kwargs
        (true_input, _) = inputs
        return true_input

    def compute_output_shape(self, input_shape):
        (input_shape, _) = input_shape
        return input_shape


def _make_inputs(input_signature):
    """Initializes keras.Input layers for a given signature.

    Args:
        input_signature (pytree of TensorSignatures): Input signature.

    Returns:
        Pytree of tf.keras.Input layers.
    """
    def init_layer(signature):
        return keras.Input(shape=signature.shape, dtype=signature.dtype)
    return data.nested_map(init_layer, input_signature)


def _make_output_heads(hidden, output_signature, output_activation, zero_init):
    """Initializes Dense layers for heads.

    Args:
        hidden (tf.Tensor or pytree of tf.Tensors): Output of the last hidden
            layer.
        output_signature (pytree of TensorSignatures): Output signature.
        output_activation (pytree of activations): Activation of every head. See
            tf.keras.layers.Activation docstring for possible values.
        zero_init (bool): Whether to initialize the heads with zeros. Useful for
            ensuring proper exploration in the initial stages of RL training.

    Returns:
        Pair (heads, masks), where heads is a pytree of head output tensors and
        masks is a pytree of keras.Input layers for passing the masks to the
        model.
    """
    # Masks are only needed for computing the loss, so we add them to the output
    # of the model. They still need to be passed as input to the model, so we
    # return the appropriate Input layer from this function, so it can be later
    # added to the inputs.
    masks = _make_inputs(output_signature)

    def init_head(layer, signature, activation, mask):
        assert signature.dtype == np.float32
        depth = signature.shape[-1]
        kwargs = {'activation': activation}
        if zero_init:
            kwargs['kernel_initializer'] = 'zeros'
            kwargs['bias_initializer'] = 'zeros'
        head = keras.layers.Dense(depth, **kwargs)(layer)
        return AddMask()((head, mask))

    if tf.is_tensor(hidden):
        hidden = data.nested_map(lambda _: hidden, output_signature)

    heads = data.nested_zip_with(
        init_head, (hidden, output_signature, output_activation, masks)
    )
    return (heads, masks)


@gin.configurable
def mlp(
    network_signature,
    hidden_sizes=(32,),
    activation='relu',
    output_activation=None,
    output_zero_init=False,
):
    """Simple multilayer perceptron."""
    inputs = _make_inputs(network_signature.input)

    x = inputs
    for h in hidden_sizes:
        x = keras.layers.Dense(h, activation=activation)(x)

    (outputs, masks) = _make_output_heads(
        x, network_signature.output, output_activation, output_zero_init
    )
    return keras.Model(inputs=(inputs, masks), outputs=outputs)


@gin.configurable
def additive_injector(primary, auxiliary):
    """Injects auxiliary input by adding normalized vectors together."""
    primary = keras.layers.LayerNormalization()(primary)
    auxiliary = keras.layers.LayerNormalization(center=False)(auxiliary)
    return keras.layers.Add()((primary, auxiliary))


def _inject_auxiliary_input(primary, auxiliary, injector):
    if injector is not None:
        depth = primary.shape[-1]
        auxiliary = keras.layers.Dense(depth)(auxiliary)
        primary = injector(primary, auxiliary)
    return primary


@gin.configurable
def convnet_mnist(
    network_signature,
    n_conv_layers=5,
    d_conv=64,
    d_ff=128,
    activation='relu',
    aux_input_injector=None,
    output_activation=None,
    output_zero_init=False,
    global_average_pooling=False,
    strides=(1, 1),
):
    """Simple convolutional network."""
    inputs = _make_inputs(network_signature.input)

    if aux_input_injector is None:
        x = inputs
        aux_input = None
    else:
        (x, aux_input) = inputs

    for _ in range(n_conv_layers):
        x = keras.layers.Conv2D(
            d_conv,
            kernel_size=(3, 3),
            padding='same',
            activation=activation,
            strides=strides,
        )(x)
    if global_average_pooling:
        x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(d_ff)(x)
    x = _inject_auxiliary_input(x, aux_input, aux_input_injector)
    x = keras.layers.Activation(activation)(x)

    (outputs, masks) = _make_output_heads(
        x, network_signature.output, output_activation, output_zero_init
    )
    return keras.Model(inputs=(inputs, masks), outputs=outputs)


def _spread_action_over_the_board(obs, action):
    """Spreads action all over the board.

    Extends channel dimension of the observation with a one-hot encoded action.

    Args:
        obs (tf.Tensor): 4D tensor of observations.
            Shape: (batch_size, height, width, n_channels)
        action (tf.Tensor): 2D tensor of one-hot encoded actions.
            Shape: (batch_size, action_space_size)

    Returns:
        4D tensor of shape:
        (batch_size, height, width, n_channels + action_space_size)
    """

    assert len(obs.shape) == 4
    assert len(action.shape) == 2

    n_actions = action.shape[-1]
    action_shape = tf.constant((-1, 1, 1, n_actions), dtype=tf.int32)
    action = tf.reshape(action, action_shape)

    multipliers = [1, obs.shape[1], obs.shape[2], 1]
    action = tf.tile(action, tf.constant(multipliers, dtype=tf.int32))

    return tf.concat([obs, action], axis=-1)


@gin.configurable
def fcn_for_env_model(
    network_signature,
    cnn_channels=64,
    cnn_n_layers=2,
    cnn_kernel_size=(5, 5),
    cnn_strides=(1, 1),
    cnn_final_pool_size=(1, 1),
    output_activation=None,
    batch_norm=False,
    output_zero_init=False,
):
    """Fully-convolutional network for trainable RL models.

    Takes (observation, action), predicts (next_observation, reward, done).
    """
    inputs = _make_inputs(network_signature.input)

    observation = inputs['observation']
    action = inputs['action']

    x = _spread_action_over_the_board(observation, action)

    for _ in range(cnn_n_layers):
        x = keras.layers.Conv2D(
            cnn_channels, kernel_size=cnn_kernel_size, strides=cnn_strides,
            padding='same', activation='relu'
        )(x)
        if batch_norm:
            x = keras.layers.BatchNormalization()(x)

    x = keras.layers.MaxPooling2D(pool_size=cnn_final_pool_size)(x)

    avg_channels = keras.layers.GlobalAveragePooling2D()(x)

    if output_activation is None:
        output_activation = {
            'next_observation': keras.activations.softmax,
            'reward': None,
            'done': keras.activations.sigmoid
        }

    final_layers = {
        'next_observation': x,
        'reward': avg_channels,
        'done': avg_channels,
    }

    (outputs, masks) = _make_output_heads(
        final_layers, network_signature.output, output_activation,
        output_zero_init
    )
    return keras.Model(inputs=(inputs, masks), outputs=outputs)


class KerasNetwork(core.TrainableNetwork):
    """TrainableNetwork implementation in Keras.

    Args:
        network_signature (NetworkSignature): Network signature.
        model_fn (callable): Function network_signature -> tf.keras.Model.
        optimizer: See tf.keras.Model.compile docstring for possible values.
        loss: See tf.keras.Model.compile docstring for possible values.
        loss_weights (list or None): Weights assigned to losses, or None if
            there's just one loss.
        weight_decay (float): Weight decay to apply to parameters.
        metrics: See tf.keras.Model.compile docstring for possible values
            (Default: None).
        train_callbacks: List of keras.callbacks.Callback instances. List of
            callbacks to apply during training (Default: None)
        seed: Seed for network initialization and other TF random operations.
            By default, use a different seed in each run.
        **compile_kwargs: These arguments are passed to tf.keras.Model.compile.
    """

    def __init__(
        self,
        network_signature,
        model_fn=mlp,
        optimizer='adam',
        loss='mean_squared_error',
        loss_weights=None,
        weight_decay=0.0,
        metrics=None,
        train_callbacks=None,
        seed=None,
        **compile_kwargs
    ):
        super().__init__(network_signature)
        self._network_signature = network_signature
        self._model = model_fn(network_signature)
        self._add_weight_decay(self._model, weight_decay)

        if seed is not None:
            tf.random.set_seed(seed)

        metrics = metrics or []
        (loss, metrics) = data.nested_map(_wrap_loss, (loss, metrics))
        self._model.compile(
            optimizer=optimizer,
            loss=loss,
            loss_weights=loss_weights,
            metrics=metrics,
            **compile_kwargs
        )

        self.train_callbacks = train_callbacks or []

    @staticmethod
    def _add_weight_decay(model, weight_decay):
        # Add weight decay in form of an auxiliary loss for every layer,
        # assuming that the weights to be regularized are in the "kernel" field
        # of every layer (true for dense and convolutional layers). This is
        # a bit hacky, but still better than having to add those losses manually
        # in every defined model_fn.
        for layer in model.layers:
            if hasattr(layer, 'kernel'):
                # Keras expects a parameterless function here. We use
                # functools.partial instead of a lambda to workaround Python's
                # late binding in closures.
                layer.add_loss(functools.partial(
                    keras.regularizers.l2(weight_decay), layer.kernel
                ))

    def train(self, data_stream, n_steps):
        """Performs one epoch of training on data prepared by the Trainer.

        Args:
            data_stream: (iterable) Python generator of triples
                (input, target, weight) to run the updates on.
            n_steps: (int) Number of training steps in the epoch.

        Returns:
            dict: Collected metrics, indexed by name.
        """
        def masked_data_stream():
            for (inp, target, mask) in data_stream():
                # Group the mask into the input.
                yield ((inp, mask), target)

        def dtypes(tensors):
            return data.nested_map(lambda x: x.dtype, tensors)

        def shapes(tensors):
            return data.nested_map(lambda x: x.shape, tensors)

        dataset = tf.data.Dataset.from_generator(
            generator=masked_data_stream,
            output_types=dtypes((self._model.input, self._model.output)),
            output_shapes=shapes((self._model.input, self._model.output)),
        )

        # WA for bug: https://github.com/tensorflow/tensorflow/issues/32912
        history = self._model.fit(
            dataset, epochs=1, verbose=0, steps_per_epoch=n_steps,
            callbacks=self.train_callbacks
        )
        # history contains epoch-indexed sequences. We run only one epoch, so
        # we take the only element.
        return {name: values[0] for (name, values) in history.history.items()}

    def predict(self, inputs):
        """Returns the prediction for a given input.

        Args:
            inputs: (Agent-dependent) Batch of inputs to run prediction on.

        Returns:
            Agent-dependent: Network predictions.
        """
        some_leaf_shape = data.choose_leaf(inputs).shape  # pylint: disable=no-member
        assert some_leaf_shape, 'KerasNetwork only works on batched inputs.'
        batch_size = some_leaf_shape[0]

        # Add dummy masks to the input.
        def one_array(signature):
            return np.ones(
                shape=((batch_size,) + signature.shape), dtype=signature.dtype
            )
        masks = data.nested_map(one_array, self._network_signature.output)

        return self._model.predict_on_batch((inputs, masks))

    @property
    def params(self):
        """Returns network parameters."""

        return self._model.get_weights()

    @params.setter
    def params(self, new_params):
        """Sets network parameters."""

        self._model.set_weights(new_params)

    def save(self, checkpoint_path):
        """Saves network parameters to a file."""

        self._model.save_weights(checkpoint_path, save_format='h5')

    def restore(self, checkpoint_path):
        """Restores network parameters from a file."""

        self._model.load_weights(checkpoint_path)
