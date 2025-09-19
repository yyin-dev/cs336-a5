import math
import torch
from typing import Callable, Literal
from einops import rearrange


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Args:
        reward_fn: scores rollout against ground truth. The dict should contain
          "reward", "format reward", "answer reward"
        rollout_response: length is rollout_batch_size = n_prompts_per_rollout_batch * group_size
        repeated_groud_truths: length is rollout_batch_size. The groud truth for
          each example is repeated group_size times.
        group_size
        advantage_eps
        normalize_by_std: if True, divide by the per-group standaard deviation; o/ only subtract
          the group mean

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]
            advantages: shape (rollout_batch_size,)
            raw_rewards: shape (rollout_batch_size,)
            metadata: other stat we choose to log (e.g. min, max, mean, std)
    """
    raw_rewards_list = [
        reward_fn(resp, gt)
        for (resp, gt) in zip(rollout_responses, repeated_ground_truths)
    ]

    raw_rewards = torch.Tensor([r["reward"] for r in raw_rewards_list])

    grouped_raw_rewards = rearrange(
        raw_rewards, "(g group_size) -> g group_size", group_size=group_size
    )
    group_mean_reward = torch.mean(grouped_raw_rewards, dim=1, keepdim=True)
    grouped_advantages = grouped_raw_rewards - group_mean_reward

    if normalize_by_std:
        group_std = torch.std(grouped_raw_rewards, dim=1, keepdim=True)
        grouped_advantages /= group_std + advantage_eps

    advantages = rearrange(
        grouped_advantages, "g group_size -> (g group_size)", group_size=group_size
    )

    # Other stats
    reward_mean = torch.mean(raw_rewards).item()

    return advantages, raw_rewards, {"mean": reward_mean}


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor, policy_log_probs: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        raw_rewards_or_advantages: (batch_size, 1)
        policy_log_probs: (batch_size, sequence_length)

    Returns:
        Shape (batch_size, sequence_length). Per-token policy-gradient loss (to
        be aggregated across the batch and sequence dimension in the training loop).
    """

    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:
        advantages: (batch_size, 1), per-example advantage
        policy_log_probs: (batch_size, seq_len), per-token log probs from the policy being trained
        old_log_probs: (batch_size, seq_len), per-token log probs from the old policy
        cliprange

    Returns:
        loss: (batch_size, seq_len), per-token clipped loss
        metadata: dict containing whatever you want to log. Suggestions:
            - Whether each token was clipped or not
    """
    # Important: Compute weight based on RAW probabilities, instead of log probs!
    weight = torch.exp(policy_log_probs - old_log_probs)

    loss = -torch.min(
        weight * advantages,
        torch.clip(weight, 1 - cliprange, 1 + cliprange) * advantages,
    )

    return loss, {}


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.

    Args:
        policy_log_probs: (B, seq_len)
        loss_type
        raw_rewards: required if loss_type == "no_baseline", (B, 1)
        advantages: required for "reinforce_with_baseline" and "grpo_clip", (B, 1)
        old_log_probs: required for "grpo_clip", shape (B, seq_len)
        clip_range: required for "grpo_clip"

    Returns;
        loss
        metadata
    """

    if loss_type == "no_baseline":
        assert raw_rewards is not None
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {}

    if loss_type == "reinforce_with_baseline":
        assert advantages is not None
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, {}

    if loss_type == "grpo_clip":
        assert advantages is not None
        assert old_log_probs is not None
        assert cliprange is not None
        loss, metadata = compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )
        return loss, metadata

    raise NotImplementedError


def masked_mean(
    tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None
) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: the tensor to compute the mean of.
        mask: same shape as tensor. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        shape matches tensor.mean(dim) semantics
    """
    sum = torch.sum(tensor * mask, dim=dim)
    cnt = torch.sum(mask, dim=dim)
    return sum / cnt


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: shape (batch_size, sequence_length): the log-probs of the policy.
        response_mask: shape (batch_size, sequence_length): the mask for the response.
        gradient_accumulation_steps
        loss_type
        raw_rewards: Needed for loss_type="no_baseline". (batch_size, 1)
        advantages:  Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
          (batch_size, 1)
        old_log_probs: Needed for loss_type="grpo_clip". (batch_size, seq_len)
        cliprange: Needed for loss_type="grpo_clip".

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            the policy gradient loss and its metadata.
    """

    # loss: (B, seq_len)
    loss, medatadata = compute_policy_gradient_loss(
        policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange
    )

    loss = masked_mean(loss, response_mask) / gradient_accumulation_steps

    loss.backward()

    return loss, medatadata
