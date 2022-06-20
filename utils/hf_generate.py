import collections

import gin
import torch

from utils import hf as hf_utils


GenerationInput = collections.namedtuple('GenerationInput', [
    'input_ids', 'attention_mask'
])


def postprocess_tensor(tensor, batch_size, num_return_sequences):
    tensor = tensor.cpu()
    assert tensor.size(0) == batch_size * num_return_sequences
    return torch.reshape(tensor, (batch_size, num_return_sequences, -1))


def generate_sequences(
        model, inputs, num_return_sequences, num_beams, max_length,
        length_penalty=1.0, temperature=1.0,
):
    """Generate sequences using beam search.

    Args:
        model: Trained transformer model.
        inputs (list of GenerationInput): Batch of input sequences for generation.
        num_return_sequences (int): Number of sequences to generate per each
            input sequence.
        num_beams (int): Number of beams in beam search.
        max_length (int): Maximum length of generated sequence. Generation stops
            at this length, if model didn't finish earlier.
        length_penalty (float): Adjust beam search score as follows:
            sequence_score = log(sequence_prob) / (sequence_len ** length_penalty)
        temperature (float): Softmax temperature.

    Returns:
        Tuple consisting of 2 torch tensors:
            Sequences: token_ids of generated sequences
                shape: (len(inputs), num_return_sequences, max_generated_seq_length)
            Scores: beam search score for each sequence
                shape: (len(inputs), num_return_sequences, 1)
    """
    assert num_return_sequences <= num_beams

    model_output = model.generate(
        input_ids=torch.tensor(
            [input.input_ids for input in inputs],
            dtype=torch.int64,
            device=model.device,
        ),
        attention_mask=torch.tensor(
            [input.attention_mask for input in inputs],
            dtype=torch.int64,
            device=model.device,
        ),
        num_return_sequences=num_return_sequences,
        max_length=max_length,
        min_length=0,

        decoder_start_token_id=(
            hf_utils.GENERATION_START_TOKEN_ID[type(model).__name__]
        ),

        do_sample=False,
        num_beams=num_beams,
        num_beam_groups=1,  # Maybe use higher values.

        # Softmax temperature. The higher the more diverse
        # are generated tokens.
        temperature=temperature,
        top_k=1000,  # 1000 > vocab_size
        top_p=1.,
        length_penalty=length_penalty,

        return_dict_in_generate=True,
        output_scores=True,
        # The following kwargs me be relevant - we may consider them in the future.
        # pad_token_id, bos_token_id, eos_token_id, decoder_start_token_id
        # forced_bos_token_id, forced_eos_token_id
    )

    return tuple(
        postprocess_tensor(
            tensor, batch_size=len(inputs),
            num_return_sequences=num_return_sequences
        )
        for tensor in [
            model_output.sequences, model_output.sequences_scores
        ]
    )


def sample_sequences(model, inputs, num_return_sequences, max_length, temperature=1.0):
    """Generate sequences using sampling.

    Args:
        model: Trained transformer model.
        inputs (list of GenerationInput): Batch of input sequences for generation.
        num_return_sequences (int): Number of sequences to generate per each
            input sequence.
        max_length (int): Maximum length of generated sequence. Generation stops
            at this length, if model didn't finish earlier.
        temperature (float): Softmax temperature.

    Returns:
        Torch tensor of token_ids - generated sequences
            shape: (len(inputs), num_return_sequences, max_generated_seq_length)
    """
    model_output = model.generate(
        input_ids=torch.tensor(
            [input.input_ids for input in inputs],
            dtype=torch.int64,
            device=model.device,
        ),
        attention_mask=torch.tensor(
            [input.attention_mask for input in inputs],
            dtype=torch.int64,
            device=model.device,
        ),
        num_return_sequences=num_return_sequences,
        max_length=max_length,
        min_length=0,

        decoder_start_token_id=(
            hf_utils.GENERATION_START_TOKEN_ID[type(model).__name__]
        ),

        do_sample=True,
        num_beams=1,
        num_beam_groups=1,

        # Softmax temperature. The higher the more diverse
        # are generated tokens.
        temperature=temperature,
        top_k=1000,  # 1000 > vocab_size
        top_p=1.,

        return_dict_in_generate=True,
        output_scores=True,
    )
    return postprocess_tensor(
        model_output.sequences,
        batch_size=len(inputs),
        num_return_sequences=num_return_sequences
    )


def compute_probabilities(beam_search_scores, sequence_lengths, length_penalty):
    assert beam_search_scores.size(0) == beam_search_scores.numel() \
           == len(sequence_lengths)
    beam_search_scores = beam_search_scores.reshape((-1,))

    # Beam search does normalization by sequence length.
    log_probs = beam_search_scores * (torch.Tensor(sequence_lengths) ** length_penalty)
    probs = torch.exp(log_probs)
    normalized_probs = probs / torch.sum(probs)
    return normalized_probs
