import math
import torch
from typing import Callable
from einops import rearrange


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
):
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
