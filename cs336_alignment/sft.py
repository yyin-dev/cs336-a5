import torch
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.generic import PaddingStrategy


def tokenize_prompt_and_output(
    prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer
) -> dict[str, torch.Tensor]:
    """
    Tokenize the promopt and output strings, and construct a mask that's 1 for
    response tokens and 0 for other tokens (prompt or padding)

    Returns: dist[str, torch.Tensor]
    """
    prompt_encoding = tokenizer(
        text=prompt_strs,
        padding=PaddingStrategy.DO_NOT_PAD,
        return_attention_mask=False,
    )
    response_encoding = tokenizer(
        text=output_strs,
        padding=PaddingStrategy.DO_NOT_PAD,
        return_attention_mask=False,
    )
    encoded_prompts = prompt_encoding.input_ids
    encoded_responses = response_encoding.input_ids
    assert len(encoded_prompts) == len(encoded_responses)

    # Use tokenizer.pad so we don't need to use [tokenizer.pad_token_id]
    encoded_prompt_and_responses = tokenizer.pad(
        [{"input_ids": p + r} for p, r in zip(encoded_prompts, encoded_responses)],
        padding=PaddingStrategy.LONGEST,
        return_tensors="pt",
    ).input_ids

    # Using torch.nn.utils.rnn.pad_sequence
    # encoded_prompt_and_responses = torch.nn.utils.rnn.pad_sequence(
    #     [torch.tensor(p + r) for (p, r) in zip(encoded_prompts, encoded_responses)],
    #     batch_first=True,
    #     padding_value=tokenizer.pad_token_id,
    # )

    prompt_lens = [len(p) for p in encoded_prompts]
    response_lens = [len(r) for r in encoded_responses]

    prompt_and_response_lens = [
        (prompt_len + response_len)
        for prompt_len, response_len in zip(prompt_lens, response_lens)
    ]
    max_len = max(prompt_and_response_lens)

    B = len(prompt_strs)
    response_mask = torch.arange(max_len).reshape([1, max_len]).repeat(B, 1)
    response_start = torch.tensor(prompt_lens).reshape((B, 1))
    response_lens = torch.tensor(response_lens).reshape((B, 1))
    response_end = response_start + response_lens
    response_mask = (response_start <= response_mask) & (response_mask < response_end)
    response_mask = response_mask.long()

    input_ids = encoded_prompt_and_responses[:, :-1]
    labels = encoded_prompt_and_responses[:, 1:]
    response_mask = response_mask[:, 1:]

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}
