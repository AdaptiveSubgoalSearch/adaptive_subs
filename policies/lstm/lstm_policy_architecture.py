from keras.layers import Masking
import keras
import numpy as np

def perfect_sequence(y_true, y_pred):
    return keras.backend.mean(
        keras.backend.min(
            keras.backend.cast(
                keras.backend.equal(
                    keras.backend.argmax(y_true, axis=-1),
                    keras.backend.argmax(y_pred, axis=-1)
                ),
                'float32'
            ),
            axis=-1
        )
    )

def accuracy_ignore_class(y_true_ohe, y_pred_logits, class_to_ignore):
    y_true_class = keras.backend.argmax(y_true_ohe)
    y_pred_class = keras.backend.argmax(y_pred_logits)
    not_ignored = keras.backend.cast(
        keras.backend.not_equal(y_pred_class, class_to_ignore), 'int32'
    )
    matches = keras.backend.cast(
        keras.backend.equal(y_true_class, y_pred_class), 'int32'
    ) * not_ignored
    accuracy = (
            keras.backend.sum(matches) /
            keras.backend.maximum(keras.backend.sum(not_ignored), 1)
    )
    return accuracy


def get_function_accuracy_ignore_padding(padding_token):
    def accuracy_ignore_padding(x, y):
        return accuracy_ignore_class(x, y, padding_token)
    return accuracy_ignore_padding

def get_lstm_goal_predictor(latent_dim, token_consts):
    num_tokens = token_consts.num_tokens
    padding_token = token_consts.padding_token

    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=(None, num_tokens))
    masked_encoder_inputs = Masking(mask_value=padding_token)(
        encoder_inputs)
    # Currently it looks like a no-op - probably because of wrong usage of mask_value parameter.
    # For parameter description see: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Masking
    # Masking guide: https://www.tensorflow.org/guide/keras/masking_and_padding#masking
    # Maybe use Embedding from here?: https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
    # this would need to undo

    encoder = keras.layers.LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(masked_encoder_inputs)

    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = keras.Input(shape=(None, num_tokens))
    masked_decoder_inputs = Masking(mask_value=padding_token)(decoder_inputs)

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True,
                                     return_state=True)
    decoder_outputs, _, _ = decoder_lstm(masked_decoder_inputs,
                                         initial_state=encoder_states)

    decoder_dense = keras.layers.Dense(num_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    return keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

def get_encoder_decoder_models(token_consts, checkpoint_path=None, model=None):

    if model is None:
        model = keras.models.load_model(
            checkpoint_path,
            custom_objects={
                "accuracy_ignore_padding": get_function_accuracy_ignore_padding(
                    token_consts.padding_token
                ),
                "perfect_sequence": perfect_sequence
            }
        )

    # The way encoder and decoder are extracted from the entire model looks
    # quite hacky and prone to network modifications.

    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[4].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = keras.Model(encoder_inputs, encoder_states)

    latent_dim = encoder_outputs.shape[1]  # Not sure if this always will be correct

    decoder_inputs = model.input[1]  # input_2
    decoder_state_input_h = keras.Input(shape=(latent_dim,), name="input_3")
    decoder_state_input_c = keras.Input(shape=(latent_dim,), name="input_4")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[5]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[6]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = keras.Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
    )

    return encoder_model, decoder_model


def sample_output_sequence(input_seq, encoder_model, decoder_model, token_consts, max_output_len=250):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, token_consts.num_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, token_consts.output_start_token] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    predicted_tokens = []

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        predicted_tokens.append(sampled_token_index)

        # Exit condition: either hit max length
        # or find stop character.
        if (
                sampled_token_index == token_consts.padding_token or
                len(predicted_tokens) >= max_output_len
        ):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, token_consts.num_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h, c]

    return predicted_tokens

def compile_lstm(model, token_consts, learning_rate):
    opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(
        optimizer=opt, loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            get_function_accuracy_ignore_padding(
                token_consts.padding_token
            ),
            perfect_sequence
        ]
    )

def encode_tokenized_formula_to_seqarrays(
        tokenized_pairs, token_consts
):
    n_formulas = len(tokenized_pairs)
    x_tokenized_data, y_tokenized_data = zip(*tokenized_pairs)

    # https://keras.io/examples/nlp/lstm_seq2seq/
    max_encoder_seq_length = max(len(x) for x in x_tokenized_data)
    max_decoder_seq_length = max(len(y) for y in y_tokenized_data)

    # create one hot encoded data for training
    num_tokens = token_consts.num_tokens
    encoder_input_data = np.zeros(
        (n_formulas, max_encoder_seq_length, num_tokens),
        dtype="float32"
    )
    decoder_input_data = np.zeros(
        (n_formulas, max_decoder_seq_length, num_tokens),
        dtype="float32"
    )
    decoder_target_data = np.zeros(
        (n_formulas, max_decoder_seq_length, num_tokens),
        dtype="float32"
    )

    padding_token = token_consts.padding_token
    for i, (x, y) in enumerate(zip(x_tokenized_data, y_tokenized_data)):
        for t, char in enumerate(x):
            encoder_input_data[i, t, char] = 1.0
        encoder_input_data[i, len(x):, padding_token] = 1.0
        for t, char in enumerate(y):
            decoder_input_data[i, t, char] = 1.0
            # decoder_target_data is ahead of decoder_input_data by one timestep
            # and doesn't include the start character.
            if t > 0:
                decoder_target_data[i, t - 1, char] = 1.0
        decoder_input_data[i, len(y):, padding_token] = 1.0
        decoder_target_data[i, len(y) - 1:, padding_token] = 1.0

    return encoder_input_data, decoder_input_data, decoder_target_data

def state_destination_formula_to_encoder_input(formula, representation):
    num_tokens = representation.token_consts.num_tokens
    tokenized_formula = representation.tokenize_formula(formula)
    encoder_input_data = np.zeros(
        (1, len(formula), num_tokens),
        dtype="float32"
    )
    for t, char in enumerate(tokenized_formula):
        encoder_input_data[0, t, char] = 1.0

    return encoder_input_data



