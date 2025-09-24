import torch
import gc
import random
import numpy as np
from torch.nn.utils import clip_grad_norm_
from sft import get_response_log_probs, tokenize_prompt_and_output
from grpo import grpo_microbatch_train_step, compute_group_normalized_rewards
from math_baseline import evaluate_vllm
from common import (
    get_device,
    CosineWarmupScheduler,
    init_vllm,
    load_policy_into_vllm_instance,
    load_math_train_using_r1_zero_prompt,
    load_math_test_using_r1_zero_prompt,
    sample,
    ceiling_dev,
)
from einops import rearrange
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
import typer
from typing_extensions import Annotated
import enum
from vllm import LLM, SamplingParams
import logging
import wandb
from drgrpo_grader import r1_zero_reward_fn
from dataclasses import dataclass, field

QWEN = "Qwen/Qwen2.5-Math-1.5B"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

app = typer.Typer(pretty_exceptions_enable=False)

# Configure basic logging to console
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class LossType(str, enum.Enum):
    NO_BASELINE = "no_baseline"
    REINFORCE_WITH_BASELINE = "reinforce_with_baseline"
    GRPO_CLIP = "grpo_clip"


@dataclass
class GrpoStepMetrics:
    step_losses: list = field(default_factory=list)
    step_entropies: list = field(default_factory=list)
    step_grad_norms: list = field(default_factory=list)
    step_clip_fractions: list = field(default_factory=list)

    def update_train_step(
        self,
        loss: torch.Tensor,
        per_token_entropy: torch.Tensor,
        grad_norm: torch.Tensor,
        clip_fraction: torch.Tensor | None = None,
    ):
        with torch.no_grad():
            self.step_losses.append(loss.item())
            self.step_entropies.append(per_token_entropy.item())
            self.step_grad_norms.append(grad_norm.item())
            if clip_fraction is not None:
                self.step_clip_fractions.append(clip_fraction.item())

    def after_optimizer_step(self):
        avg_loss = (
            sum(self.step_losses) / len(self.step_losses) if self.step_losses else 0
        )
        avg_entropy = (
            sum(self.step_entropies) / len(self.step_entropies)
            if self.step_entropies
            else 0
        )
        avg_grad_norm = (
            sum(self.step_grad_norms) / len(self.step_grad_norms)
            if self.step_grad_norms
            else 0
        )
        avg_clip_fraction = (
            sum(self.step_clip_fractions) / len(self.step_clip_fractions)
            if self.step_clip_fractions
            else None
        )

        return avg_loss, avg_entropy, avg_grad_norm, avg_clip_fraction


@app.command()
def main(
    train_dataset: Annotated[str, typer.Option()],
    test_dataset: Annotated[str, typer.Option()],
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,  # disallow empty string responses
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,  # defaults to 1, i.e. on-policy
    train_batch_size: int = 256,  # defaults to the same as rollout_batch_size, i.e. on-policy
    train_microbatch_size: int = 2,
    gradient_accumulation_steps: int = 128,
    gpu_memory_utilization: float = 0.85,
    loss_type: LossType = LossType.REINFORCE_WITH_BASELINE,
    use_std_normalization: bool = True,
    cliprange: float = 0.2,
    eval_every_n_steps: int = 5,
    length_normalization: bool = False,
):
    assert rollout_batch_size * epochs_per_rollout_batch % train_batch_size == 0

    assert (
        train_batch_size % train_microbatch_size == 0
    ), "train_batch_size must be divisible by microbatch size"
    gradient_accumulation_steps = train_batch_size // train_microbatch_size

    assert (
        rollout_batch_size % group_size == 0
    ), "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout_batch = rollout_batch_size // group_size

    assert (
        train_batch_size >= group_size
    ), "train_batch_size must be greater than or equal to group_size"

    assert torch.cuda.is_available()

    # wandb initialization
    name = f"grpo-{loss_type.value}-rollout{rollout_batch_size}-grp{group_size}-lr{learning_rate}-epoch{epochs_per_rollout_batch}-train_bs{train_batch_size}"
    if length_normalization:
        name += "-lengthnorm"
    wandb.init(
        project="grpo-experiment",
        name=name,
        config={
            "n_grpo_steps": n_grpo_steps,
            "learning_rate": learning_rate,
            "advantage_eps": advantage_eps,
            "rollout_batch_size": rollout_batch_size,
            "group_size": group_size,
            "sampling_temperature": sampling_temperature,
            "epochs_per_rollout_batch": epochs_per_rollout_batch,
            "train_batch_size": train_batch_size,
            "train_microbatch_size": train_microbatch_size,
            "loss_type": loss_type.value,
            "use_std_normalization": use_std_normalization,
            "cliprange": cliprange,
            "length_normalization": length_normalization,
        },
    )

    # device
    train_device = get_device("train")
    eval_device = get_device("eval")
    print(f"Train device: {train_device}; Eval device: {eval_device}")

    # dataset
    train_prompt_strs, train_ground_truths = load_math_train_using_r1_zero_prompt(
        train_dataset, "ground_truth"
    )
    test_prompt_strs, test_ground_truth_strs = load_math_test_using_r1_zero_prompt(
        test_dataset
    )

    # models
    pretrained_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        QWEN,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=train_device,
    )
    model = torch.compile(pretrained_model)

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(QWEN)
    eval_model = init_vllm(QWEN, eval_device, SEED, gpu_memory_utilization)

    grpo_rollout_sampling_params = SamplingParams(
        temperature=sampling_temperature,
        min_tokens=sampling_min_tokens,
        max_tokens=sampling_max_tokens,
        n=group_size,
        seed=SEED,
        top_p=1.0,
        stop="</answer>",
        include_stop_str_in_output=True,
    )

    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop="</answer>",
        include_stop_str_in_output=True,
        seed=SEED,
    )

    N = len(train_prompt_strs)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    total_optimizer_steps = (
        rollout_batch_size * epochs_per_rollout_batch * n_grpo_steps // train_batch_size
    )
    warmup_ratio = 0.03
    warmup_steps = int(total_optimizer_steps * warmup_ratio)
    lr_scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps,
        total_optimizer_steps,
        learning_rate,
        1e-8,
    )
    logging.info(
        f"total optimizer steps: {total_optimizer_steps}, warmup_steps: {warmup_steps}"
    )

    # Outer loop
    for grpo_step in range(n_grpo_steps):
        logging.info(f"GRPO step: {grpo_step}")
        logging.info(
            f"GPU memory allocated: {torch.cuda.memory_allocated(train_device) / 1024**3:.2f} GB"
        )

        # rollout
        rollout_prompt_start_idx = (grpo_step * n_prompts_per_rollout_batch) % N
        rollout_prompt_indices = [
            (rollout_prompt_start_idx + i) % N
            for i in range(n_prompts_per_rollout_batch)
        ]
        rollout_prompts = [train_prompt_strs[i] for i in rollout_prompt_indices]
        rollout_ground_truths = [train_ground_truths[i] for i in rollout_prompt_indices]

        load_policy_into_vllm_instance(model, eval_model)
        outputs = eval_model.generate(rollout_prompts, grpo_rollout_sampling_params)

        # rewards and advanatages
        rollout_responses = [
            completion.text for output in outputs for completion in output.outputs
        ]
        repeated_prompts = [p for p in rollout_prompts for _ in range(group_size)]
        repeated_ground_truths = [
            gt for gt in rollout_ground_truths for _ in range(group_size)
        ]
        advantages, raw_rewards, reward_stat = compute_group_normalized_rewards(
            r1_zero_reward_fn,
            rollout_responses,
            repeated_ground_truths,
            group_size,
            advantage_eps,
            use_std_normalization,
        )
        advantages = advantages.to(train_device)
        raw_rewards = raw_rewards.to(train_device)

        train_tokenized = tokenize_prompt_and_output(
            repeated_prompts, rollout_responses, tokenizer
        )

        # compute old_log_probs and reuse for each epoch
        with torch.inference_mode():
            old_log_probs = []

            for i in range(ceiling_dev(rollout_batch_size, train_microbatch_size)):
                start = i * train_microbatch_size
                end = min(start + train_microbatch_size, rollout_batch_size)
                indices = torch.tensor(range(start, end))
                input_ids, labels, response_mask = sample(
                    train_tokenized["input_ids"],
                    train_tokenized["labels"],
                    train_tokenized["response_mask"],
                    indices,
                    train_device,
                )
                old_log_probs.append(
                    get_response_log_probs(
                        model, input_ids, labels, return_token_entropy=False
                    )["log_probs"]
                )

            old_log_probs = torch.cat(old_log_probs, dim=0)

        # train steps
        # We will take [epochs_per_rollout_batch] over [rollout_batch_size].
        n_train_steps = (
            rollout_batch_size * epochs_per_rollout_batch // train_microbatch_size
        )

        # collect metrics for logging
        grpo_step_metrics = GrpoStepMetrics()

        # Inner loop
        for train_step in range(n_train_steps):
            start_idx = (train_step * train_microbatch_size) % rollout_batch_size
            indices = [
                (start_idx + i) % rollout_batch_size
                for i in range(train_microbatch_size)
            ]
            input_ids, labels, response_mask = sample(
                train_tokenized["input_ids"],
                train_tokenized["labels"],
                train_tokenized["response_mask"],
                torch.tensor(indices),
                train_device,
            )

            policy_response = get_response_log_probs(
                model, input_ids, labels, return_token_entropy=True
            )
            policy_log_probs = policy_response["log_probs"]
            token_entropy = policy_response["token_entropy"]

            microbatch_raw_rewards = rearrange(
                raw_rewards[indices], "(b d) -> b d", d=1
            )
            microbatch_advantages = rearrange(advantages[indices], "(b d) -> b d", d=1)
            microbatch_old_log_probs = old_log_probs[indices]
            assert policy_log_probs.shape == microbatch_old_log_probs.shape
            loss, loss_metadata = grpo_microbatch_train_step(
                policy_log_probs,
                response_mask,
                gradient_accumulation_steps,
                loss_type.value,
                microbatch_raw_rewards,
                microbatch_advantages,
                microbatch_old_log_probs,
                cliprange,
                length_normalization,
            )

            per_token_entropy = torch.mean(token_entropy[response_mask])
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
            clip_fraction = None
            # clip fraction for off-policy
            if loss_type.value == "grpo_clip":
                weight = torch.exp(policy_log_probs - microbatch_old_log_probs)
                clipped = (weight < (1 - cliprange)) | (weight > (1 + cliprange))
                clip_fraction = torch.mean(clipped[response_mask].float())

            grpo_step_metrics.update_train_step(
                loss, per_token_entropy, grad_norm, clip_fraction
            )

            # finished a batch
            if (train_step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                # calculate global step
                num_optimizer_steps_per_grpo_step = (
                    n_train_steps // gradient_accumulation_steps
                )
                global_step = (
                    grpo_step * num_optimizer_steps_per_grpo_step
                    + train_step // gradient_accumulation_steps
                )

                # log training metrics
                (avg_loss, avg_entropy, avg_grad_norm, avg_clip_fraction) = (
                    grpo_step_metrics.after_optimizer_step()
                )
                wandb_data = {
                    "grpo_step": grpo_step,
                    "global_step": global_step,
                    "loss": avg_loss,
                    "token_entropy": avg_entropy,
                    "grad_norm": avg_grad_norm,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "train_reward_mean": reward_stat["mean"],
                }

                if avg_clip_fraction is not None:
                    wandb_data["clip_fraction"] = avg_clip_fraction

                # log
                log_msg = f"GRPO Step {grpo_step}, Global Step {global_step}: loss={avg_loss:.6f}, entropy={avg_entropy:.6f}, grad_norm={avg_grad_norm:.6f}, train_reward={reward_stat['mean']:.6f}"
                if avg_clip_fraction is not None:
                    log_msg += f", clip_fraction={avg_clip_fraction:.4f}"

                logging.info(log_msg)

                # evaluate every N steps
                if (global_step + 1) % eval_every_n_steps == 0:
                    load_policy_into_vllm_instance(model, eval_model)

                    eval_res = evaluate_vllm(
                        eval_model,
                        r1_zero_reward_fn,
                        test_prompt_strs,
                        test_ground_truth_strs,
                        eval_sampling_params,
                    )
                    reward_stat = eval_res.reward_stat
                    wandb_data["eval_reward_mean"] = reward_stat["final"]["mean"]

                    # log evaluation results
                    logging.info(
                        f"Evaluation at step {global_step}: final_reward={reward_stat['final']['mean']:.4f}, format_reward={reward_stat['format']['mean']:.4f}, answer_reward={reward_stat['answer']['mean']:.4f}"
                    )

                wandb.log(wandb_data)

        # Cleanup at end of GRPO step
        # Without this explicit deletion, the GPU memory increases after each
        # GRPO step and causes OOM eventually. It's not 100% clear to me why GC
        # failed to work here...
        del old_log_probs, train_tokenized, advantages, raw_rewards
        gc.collect()
        torch.cuda.empty_cache()
        logging.info(
            f"After cleanup - GPU memory allocated: {torch.cuda.memory_allocated(train_device) / 1024**3:.2f} GB"
        )


if __name__ == "__main__":
    app()
