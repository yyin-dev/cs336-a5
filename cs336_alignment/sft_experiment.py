"""
uv run cs336_alignment/sft_experiment.py \
    --batch-size 4 \
    --unique-examples 16 \
    --train-set ./data/math/train \
    --gradient-accumulation-steps \
    --lr 1e-4
"""

from vllm import LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from argparse import ArgumentParser
from datasets import Dataset, load_from_disk
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
import random
import wandb
import math
from sft import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
)

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)


def cosine_lr_schedule_with_warmup(t, lr_max, lr_min, T_w, T_c):
    if t < T_w:
        return t / T_w * lr_max

    if T_w <= t <= T_c:
        return lr_min + 1 / 2 * (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w))) * (
            lr_max - lr_min
        )

    # t > T_c
    return lr_min


# Custom LR scheduler using cosine_lr_schedule_with_warmup
class CosineWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, total_steps, lr_max, lr_min):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_max = lr_max
        self.lr_min = lr_min
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step):
        lr = cosine_lr_schedule_with_warmup(
            step, self.lr_max, self.lr_min, self.warmup_steps, self.total_steps
        )
        return lr / self.lr_max  # LambdaLR multiplies by base_lr, so we normalize


def init_vllm(
    model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85
) -> LLM:
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)

    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype="bfloat16",
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def get_device(mode: str):
    if torch.cuda.is_available():
        if torch.cuda.device_count() >= 2:
            if mode == "train":
                return "cuda:0"
            elif mode == "eval":
                return "cuda:1"
            else:
                raise ValueError(f"Unknown mode: {mode}")

        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def main():
    parser = ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--unique-examples", type=int, required=True)
    parser.add_argument("--train-set", type=str, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    args = parser.parse_args()

    # wandb
    run = wandb.init(
        name=f"sft-{args.unique_examples}-lr{args.lr}",
        config={
            "unique_examples": args.unique_examples,
            "lr": args.lr,
            "gradient_accum_steps": args.gradient_accumulation_steps,
        },
    )

    # device
    train_device = get_device("train")
    print(f"train_device: {train_device}")

    # model
    qwen = "Qwen/Qwen2.5-Math-1.5B"
    model = AutoModelForCausalLM.from_pretrained(
        qwen,
        torch_dtype=torch.bfloat16,
        attn_implementation=(
            "flash_attention_2" if torch.cuda.is_available() else "sdpa"
        ),
    )
    model.to(train_device)
    tokenizer = AutoTokenizer.from_pretrained(qwen)

    # dataset
    train_dataset: Dataset = load_from_disk(
        args.train_set
    )  # pyright: ignore[reportAssignmentType]
    train_dataset = train_dataset.select_columns(["problem", "solution"])
    prompt_strs = [qa["problem"] for qa in train_dataset]
    output_strs = [qa["solution"] for qa in train_dataset]
    tokenized = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
    train_inputs_ids = tokenized["input_ids"]
    train_labels = tokenized["labels"]
    train_response_mask = tokenized["response_mask"]
    microbatch_size = max(args.batch_size // args.gradient_accumulation_steps, 1)
    print(f"Microbatch size: {microbatch_size}")

    # sft loop
    num_steps = (args.unique_examples + microbatch_size - 1) // microbatch_size

    opt = AdamW(model.parameters(), lr=args.lr)

    warmup_steps = max(1, num_steps // 10)  # 10% warmup
    lr_scheduler = CosineWarmupScheduler(opt, warmup_steps, num_steps, args.lr, 1e-7)

    # TODO: log generation in the loop
    for step in range(num_steps):
        print(f"Step {step}")

        # sampling
        start = microbatch_size * step
        end = microbatch_size * (step + 1)
        input_ids = train_inputs_ids[start:end, :].to(train_device)
        labels = train_labels[start:end, :].to(train_device)
        response_mask = train_response_mask[start:end, :].to(train_device)
        print(input_ids.shape, labels.shape, response_mask.shape)

        print("Forward...")
        response_log_probs = get_response_log_probs(
            model,
            input_ids,
            labels,
            return_token_entropy=False,
        )
        # TODO: report entropy
        policy_log_probs = response_log_probs["log_probs"]

        print("loss and backward...")
        loss, stat = sft_microbatch_train_step(
            policy_log_probs,
            response_mask,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            normalize_constant=1.0,
        )

        print(f"Loss: {loss.item()}, stat: {stat}")

        clip_grad_norm_(model.parameters(), max_norm=1.0)

        if (step + 1) % args.gradient_accumulation_steps == 0:
            print("optimizer step...")
            opt.step()
            opt.zero_grad()
            lr_scheduler.step()

        if (step + 1) % (4 * args.gradient_accumulation_steps) == 0:
            # TODO: evaluation every few batches using vLLM
            pass

        wandb_data = {
            "step": step,
            "lr": lr_scheduler.get_last_lr()[0],
            "loss": loss.item(),
        }
        wandb.log(wandb_data)

        # TODO: save checkpoint


if __name__ == "__main__":
    main()
