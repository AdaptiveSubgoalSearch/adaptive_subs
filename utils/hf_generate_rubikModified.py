import collections

import torch


GenerationInput = collections.namedtuple('GenerationInput', [
    'input_ids', 'attention_mask'
])


def generate_sequences(model, inputs, num_return_sequences, num_beams, max_length):
    """Generate sequences using beam search.

    Args:
        model: Trained transformer model.
        inputs (list of GenerationInput): Batch of input sequences for generation.
        num_return_sequences (int): Number of sequences to generate per each
            input sequence.
        num_beams (int): Number of beams in beam search.
        max_length (int): Maximum length of generated sequence. Generation stops
            at this length, if model didn't finish earlier.

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
        # attention_mask=torch.tensor(
        #     [input.attention_mask for input in inputs],
        #     dtype=torch.int64,
        #     device=model.device,
        # ),
        # num_return_sequences=num_return_sequences,
        max_length=max_length,
        # min_length=0,

        decoder_start_token_id=2,  # eos_token_id

        # do_sample=False,
        # num_beams=num_beams,
        # num_beam_groups=1,  # Maybe use higher values.

        # Softmax temperature. The higher the more diverse
        # are generated tokens.
        # temperature=1.0,
        # top_k=1000,  # 1000 > vocab_size
        # top_p=1.,

        # return_dict_in_generate=True,
        # output_scores=True,
        # The following kwargs me be relevant - we may consider them in the future.
        # pad_token_id, bos_token_id, eos_token_id, decoder_start_token_id
        # forced_bos_token_id, forced_eos_token_id
    )

    result = []
    for tensor in [model_output.sequences]:
        tensor = tensor.cpu()
        assert tensor.size(0) == len(inputs) * num_return_sequences
        result.append(torch.reshape(tensor, (len(inputs), num_return_sequences, -1)))
    return result[0]
