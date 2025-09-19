"""
uv run cs336_alignment/expert_iteration_experiment.py \
  --expert-iteration-batch-size 512 \
  --rollout 5 \
  --batch-size 8 \
  --microbatch-size 2 \
  --train-set ./data/math/train \
  --test-set ./data/math/test \
  --epochs 2 \
  --lr 5e-5
"""

from vllm import SamplingParams, LLM
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from argparse import ArgumentParser
from datasets import Dataset, load_from_disk
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
import random
import statistics
import wandb
from typing import Any
from sft import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
)
from math_baseline import evaluate_vllm
from drgrpo_grader import r1_zero_reward_fn, extract_answer, grade
from common import (
    get_device,
    ceiling_dev,
    CosineWarmupScheduler,
    load_policy_into_vllm_instance,
    init_vllm,
    load_math_train_using_r1_zero_prompt,
    load_math_test_using_r1_zero_prompt,
    sample,
    fit_r1_zero_question,
)

QWEN = "Qwen/Qwen2.5-Math-1.5B"
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)


def sft(
    train_prompt_strs: list[str],
    train_output_strs: list[str],
    test_prompt_strs: list[str],
    test_ground_truth_strs: list[str],
    epochs: int,
    batch_size: int,
    microbatch_size: int,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_model: LLM,
    eval_sampling_params: SamplingParams,
    lr: float,
    train_device: str,
) -> dict[str, Any]:
    N = len(train_output_strs)
    print(f"SFT, N={N}, epochs={epochs}")
    gradient_accumulation_steps = batch_size // microbatch_size

    train_tokenized = tokenize_prompt_and_output(
        train_prompt_strs, train_output_strs, tokenizer
    )

    num_steps_per_epoch = ceiling_dev(N, microbatch_size)
    num_steps = epochs * num_steps_per_epoch
    num_optimizer_steps = epochs * ceiling_dev(N, batch_size)

    opt = AdamW(model.parameters(), lr=lr)
    warmup_steps = max(1, int(num_optimizer_steps * 0.05))
    lr_scheduler = CosineWarmupScheduler(
        opt, warmup_steps, num_optimizer_steps, lr, 1e-7
    )

    losses = []
    entropies = []
    for step in range(num_steps):
        start_idx = (step * microbatch_size) % N
        indices = [(start_idx + i) % N for i in range(microbatch_size)]
        input_ids, labels, response_mask = sample(
            train_tokenized["input_ids"],
            train_tokenized["labels"],
            train_tokenized["response_mask"],
            torch.tensor(indices),
            train_device,
        )

        response_log_probs = get_response_log_probs(
            model, input_ids, labels, return_token_entropy=True
        )

        policy_log_probs = response_log_probs["log_probs"]
        entropy: torch.Tensor = response_log_probs["token_entropy"]

        loss, stat = sft_microbatch_train_step(
            policy_log_probs,
            response_mask,
            gradient_accumulation_steps,
            normalize_constant=1.0,
        )

        clip_grad_norm_(model.parameters(), max_norm=1.0)

        with torch.no_grad():
            # The `loss` here is (loss from microbatch / batch size).
            losses.append(loss.item())
            # The `entropy` here is (total entropy from microbatch),
            # (B, sequence_length).
            # The response_mask is (B, sequence_length) too.

            per_token_entropy = torch.mean(entropy[response_mask])
            entropies.append(per_token_entropy.item())

        if (step + 1) % gradient_accumulation_steps == 0:
            opt.step()
            opt.zero_grad()
            lr_scheduler.step()

    wandb_data = {}
    wandb_data["loss"] = statistics.mean(losses)
    wandb_data["entropy"] = statistics.mean(entropies)

    # evaluate on validation set once after all SFT steps
    load_policy_into_vllm_instance(model, eval_model)

    eval_res = evaluate_vllm(
        eval_model,
        r1_zero_reward_fn,
        test_prompt_strs,
        test_ground_truth_strs,
        eval_sampling_params,
    )
    reward_stat = eval_res.reward_stat
    print(reward_stat)

    return {"wandb": wandb_data, "reward": reward_stat}


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-eibs",
        "--expert-iteration-batch-size",
        type=int,
        required=True,
        choices=[512, 1024, 2048],
    )
    parser.add_argument("-G", "--rollout", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--microbatch-size", type=int, required=True)
    parser.add_argument("--train-set", type=str, required=True)
    parser.add_argument("--test-set", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True, help="SFT epochs")
    parser.add_argument("--lr", type=float, required=True)
    args = parser.parse_args()

    ei_batch_size: int = args.expert_iteration_batch_size
    G: int = args.rollout
    batch_size: int = args.batch_size
    microbatch_size: int = args.microbatch_size
    train_set: str = args.train_set
    test_set: str = args.test_set
    epochs: int = args.epochs
    lr: float = args.lr

    # validate args
    if batch_size % microbatch_size != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by microbatch_size ({microbatch_size})"
        )

    gradient_accumulation_steps = batch_size // microbatch_size
    print(
        f"Batch size: {batch_size}, microbatch size: {microbatch_size}, gradient accum. steps: {gradient_accumulation_steps}"
    )

    # wandb
    name = f"ei-eibs{ei_batch_size}-G{G}-ep{epochs}-lr{lr}-bs{batch_size}"
    run = wandb.init(
        project="expert-iteration",
        name=name,
        config={
            "EI_batch_size": ei_batch_size,
            "G": G,
            "SFT_batch_size": batch_size,
            "SFT_epoch": epochs,
            "SFT_lr": lr,
        },
    )

    # device
    train_device = get_device("train")
    eval_device = get_device("eval")
    print(f"Train device: {train_device}; Eval device: {eval_device}")

    # model
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        QWEN,
        torch_dtype=torch.bfloat16,
        attn_implementation=(
            "flash_attention_2" if torch.cuda.is_available() else "sdpa"
        ),
        device_map=train_device,
    )
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(QWEN)

    eval_model = None
    if torch.cuda.is_available():
        eval_model = init_vllm(QWEN, eval_device, SEED)

    sampling_temperature = 1.0
    sampling_min_tokens = 4
    sampling_max_tokens = 1024
    ei_sampling_params = SamplingParams(
        temperature=sampling_temperature,
        min_tokens=sampling_min_tokens,
        max_tokens=sampling_max_tokens,
        n=G,
        seed=SEED,
        top_p=1.0,
        stop="</answer>",
        include_stop_str_in_output=True,
    )

    # dataset
    train_prompt_strs, train_ground_truths = load_math_train_using_r1_zero_prompt(
        train_set, "ground_truth"
    )

    test_prompt_strs, test_ground_truth_strs = load_math_test_using_r1_zero_prompt(
        test_set
    )

    # Expert iteration loop
    num_ei_steps = 5
    for ei_step in range(num_ei_steps):
        print(f"Expert Iteration step: {ei_step}")

        N = len(train_prompt_strs)
        start_idx = (ei_step * ei_batch_size) % N
        indices = [(start_idx + i) % N for i in range(ei_batch_size)]
        prompts = [train_prompt_strs[i] for i in indices]
        ground_truths = [train_ground_truths[i] for i in indices]

        sft_data: list[tuple[str, str]] = []
        if eval_model:
            load_policy_into_vllm_instance(model, eval_model)
            outputs = eval_model.generate(prompts, ei_sampling_params)

            for prompt, output, ground_truth in zip(prompts, outputs, ground_truths):
                for completion in output.outputs:
                    reward = r1_zero_reward_fn(completion.text, ground_truth)

                    if reward["reward"] > 0:
                        sft_data.append((prompt, completion.text))
        else:
            raise NotImplementedError

        sft_prompts = [x[0] for x in sft_data]
        sft_labels = [x[1] for x in sft_data]

        # SFT loop
        eval_sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            stop="</answer>",
            include_stop_str_in_output=True,
        )

        wandb_data = {}
        wandb_data["ei_step"] = ei_step
        ei_success_rate = (
            len(sft_data) / (ei_batch_size * G) if (ei_batch_size * G > 0) else 0
        )
        wandb_data["ei_success_rate"] = ei_success_rate
        wandb_data["sft_dataset_size"] = len(sft_data)

        if len(sft_prompts) == 0:
            print("No correct completions found, skip to next EI step")

            wandb_data["loss"] = None
            wandb_data["entropy"] = None
            wandb_data["format_mean"] = None
            wandb_data["final_mean"] = None
        else:
            sft_stat = sft(
                sft_prompts,
                sft_labels,
                test_prompt_strs,
                test_ground_truth_strs,
                epochs,
                batch_size,
                microbatch_size,
                model,
                tokenizer,
                eval_model,
                eval_sampling_params,
                lr,
                train_device,
            )

            wandb_data["loss"] = sft_stat["wandb"]["loss"]
            wandb_data["entropy"] = sft_stat["wandb"]["entropy"]
            wandb_data["format_mean"] = sft_stat["reward"]["format"]["mean"]
            wandb_data["final_mean"] = sft_stat["reward"]["final"]["mean"]

        wandb.log(wandb_data)


if __name__ == "__main__":
    main()
