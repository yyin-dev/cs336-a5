import torch
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.generic import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from einops import rearrange


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


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Args:
      logits: (batch_size, sequence_length, vocab_size)

    Returns:
       (batch_size, sequence_length)
    """
    logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
    diff = logits - logsumexp
    return -torch.sum(torch.exp(diff) * diff, dim=-1)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Args
        model
        input_ids: (batch_size, sequence_length)
        labels: (batch_size, sequence_length)
        return_token_entropy: if true, also return per-token entropy by calling compute_entropy.

    Returns
        "log_probs": shape (batch_size, sequence_length), *per-token* conditional log-probs. log(p_theta(x_t | x_<t))
        "token_entropy": optional, shape (batch_size, sequence_length), per-token entropy for each position
    """
    # log(p(x)) = log(softmax(logits)) = logits - logsumexp(logits)
    logits = model(input_ids).logits  # (b, s, V)
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    log_probs = torch.gather(
        log_probs, dim=-1, index=rearrange(labels, "b (s d) -> b s d", d=1)
    )
    log_probs = rearrange(log_probs, "b s d -> b (s d)", d=1)

    res = {"log_probs": log_probs}

    if return_token_entropy:
        entropy = compute_entropy(logits)
        res["token_entropy"] = entropy

    return res


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Input:
        tensor
        mask: same shape as [tensor]
        dim: int | None. When None, sum along all dimensions.
    """

    # tensor[mask] performs boolean indexing, and returns a 1-D tensor
    # containing selected items (even if tensor and mask are not 1D!).
    # This is equivalent to torch.masked_select
    #
    # To keep the shape of `tensor` and zero out entries where `mask` is 0,
    # there are several approaches:
    # 1. tensor * mask
    # 2. torch.where(mask, tensor, torch.zeros_like(tensor))
    # 3. torch[~mask] = 0
    # 4. tensor.masked_fill_(~mask, 0)
    #
    # It's a bit weird that tensor[mask] returns 1d, while tensor[~mask] = 0
    # works. Both are using the bracket operator?
    # Actually the two operator are not the same; one is a selection operation,
    # and the other is an assignment operation.

    tensor[~mask] = 0
    if dim is None:
        s = tensor.sum()
    else:
        s = torch.sum(tensor, dim=dim)

    s /= normalize_constant
    return s


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Input:
        policy_log_probs: (batch_size, sequence_length). Per-token log probabilities from the SFT policy being trained.
        response_mask: (batch_size, sequence_length). 1 for response tokens, 0 for prompt/padding
        gradient_accumulation_steps
        normalize_constant

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]
          loss: scalar tensor
          metadata: dict with metadata from the loss call, and any other stat we might we to log
    """

    # policy_log_probs is the log probabilities the model generates labels based on
    # prompts (see `get_response_log_probs`). Thus, we want to maximize it, so
    # the loss should be the negated sum of log probs (negative log likelihood).
    #
    # To compute the probability of a sequence, you multipy the per-token
    # probabilities. However, this is numerically unstable for long sequences
    # because multiply a sequence of numbers < 1 produces a very small number.
    # Log probability turns multiplication into sum and is numerically more stable.
    #
    # Cross-entropy loss in this case is another name for negative log likelihood.
    # Cross-entropy measures the difference between two distributions. When you
    # apply cross-entropy loss with a "groud-truth" distribution that's a one-hot
    # vector, the formula simplifies to negative log likelihood.

    B = policy_log_probs.shape[0]

    loss = (
        -policy_log_probs[response_mask].sum()
        / normalize_constant
        / gradient_accumulation_steps
        / B
    )

    loss.backward()

    return (loss, {})
