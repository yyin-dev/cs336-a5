"""
uv run cs336_alignment/sft_experiment.py \
    --batch-size 16 \
    --microbatch-size 2 \
    --unique-examples 128 \
    --epochs 10 \
    --train-set ./data/math/train \
    --test-set ./data/math/test \
    --lr 1e-4 \
    --checkpoint-dir ../checkpoints
"""

from vllm import SamplingParams
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
import wandb
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
    fit_r1_zero_question,
    fit_r1_zero_output,
    sample,
)

QWEN = "Qwen/Qwen2.5-Math-1.5B"
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)


def main():
    parser = ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--unique-examples", type=int, required=True)
    parser.add_argument("--train-set", type=str, required=True)
    parser.add_argument("--test-set", type=str, required=True)
    parser.add_argument("--microbatch-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--eval-every-n-batches", type=int, default=10)
    parser.add_argument("--filter-out-incorrect-training-data", action="store_true")
    parser.add_argument("--init-eval", action="store_true")
    args = parser.parse_args()

    # Validate arguments
    if args.batch_size % args.microbatch_size != 0:
        raise ValueError(
            f"batch_size ({args.batch_size}) must be divisible by microbatch_size ({args.microbatch_size})"
        )

    gradient_accumulation_steps = args.batch_size // args.microbatch_size
    microbatch_size = args.microbatch_size

    # wandb
    name = f"sft-{args.unique_examples}-ep{args.epochs}-lr{args.lr}-bs{args.batch_size}"
    if args.filter_out_incorrect_training_data:
        name += "-filter"

    run = wandb.init(
        name=name,
        config={
            "unique_examples": args.unique_examples,
            "epochs": args.epochs,
            "lr": args.lr,
            "gradient_accum_steps": gradient_accumulation_steps,
            "microbatch_size": microbatch_size,
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
    ).to(train_device)

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(QWEN)

    # dataset
    # train
    train_dataset: Dataset = load_from_disk(
        args.train_set
    )  # pyright: ignore[reportAssignmentType]
    train_dataset = train_dataset.select_columns(["problem", "solution", "answer"])

    # Filter out incorrect training data if requested
    if args.filter_out_incorrect_training_data:
        original_size = len(train_dataset)
        train_dataset = train_dataset.filter(
            lambda v: extract_answer(v["solution"]) is not None
            and grade(extract_answer(v["solution"]), v["answer"])
        )
        filtered_size = len(train_dataset)
        print(
            f"Filtered training data: {original_size} -> {filtered_size} examples ({filtered_size/original_size:.1%} retained)"
        )

    train_prompt_strs = [fit_r1_zero_question(qa["problem"]) for qa in train_dataset]
    train_output_strs = [
        fit_r1_zero_output(qa["solution"], qa["answer"]) for qa in train_dataset
    ]
    train_tokenized = tokenize_prompt_and_output(
        train_prompt_strs, train_output_strs, tokenizer
    )
    train_input_ids = train_tokenized["input_ids"]
    train_labels = train_tokenized["labels"]
    train_response_mask = train_tokenized["response_mask"]

    # test
    test_dataset: Dataset = load_from_disk(
        args.test_set
    )  # pyright: ignore[reportAssignmentType]
    test_dataset = test_dataset.select_columns(["problem", "answer"])
    test_prompt_strs = [fit_r1_zero_question(qa["problem"]) for qa in test_dataset]
    test_ground_truth_strs = [qa["answer"] for qa in test_dataset]

    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop="</answer>",
        include_stop_str_in_output=True,
    )

    print(
        f"Microbatch size: {microbatch_size}, Gradient accumulation steps: {gradient_accumulation_steps}"
    )

    eval_model = None
    if torch.cuda.is_available():
        eval_model = init_vllm(QWEN, eval_device, SEED)

        if args.init_eval:
            eval_res = evaluate_vllm(
                eval_model,
                r1_zero_reward_fn,
                test_prompt_strs,
                test_ground_truth_strs,
                eval_sampling_params,
            )
            eval_reward_stat = eval_res.reward_stat
            print(eval_reward_stat)

    # sft loop
    steps_per_epoch = ceiling_dev(args.unique_examples, microbatch_size)
    num_steps = steps_per_epoch * args.epochs
    num_optimizer_steps = (
        ceiling_dev(args.unique_examples, args.batch_size) * args.epochs
    )

    opt = AdamW(model.parameters(), lr=args.lr)

    warmup_steps = max(1, num_optimizer_steps // 10)  # 10% warmup
    lr_scheduler = CosineWarmupScheduler(
        opt, warmup_steps, num_optimizer_steps, args.lr, 1e-7
    )

    # TODO: log generation in the loop
    for step in range(num_steps):
        print(f"Step {step}")
        wandb_data = {}

        # deterministic sequential sampling respecting unique_examples
        max_examples = min(args.unique_examples, len(train_input_ids))
        start_idx = (step * microbatch_size) % max_examples
        indices = [(start_idx + i) % max_examples for i in range(microbatch_size)]

        index = torch.tensor(indices)
        input_ids, labels, response_mask = sample(
            train_input_ids, train_labels, train_response_mask, index, train_device
        )

        response_log_probs = get_response_log_probs(
            model,
            input_ids,
            labels,
            return_token_entropy=False,
        )
        # TODO: report entropy
        policy_log_probs = response_log_probs["log_probs"]

        loss, stat = sft_microbatch_train_step(
            policy_log_probs,
            response_mask,
            gradient_accumulation_steps=gradient_accumulation_steps,
            normalize_constant=1.0,
        )

        print(f"Loss: {loss.item()}, stat: {stat}")

        clip_grad_norm_(model.parameters(), max_norm=1.0)

        if (step + 1) % gradient_accumulation_steps == 0:
            opt.step()
            opt.zero_grad()
            lr_scheduler.step()

        # evaluate every few batches using vLLM
        eval_every_n_steps = gradient_accumulation_steps * args.eval_every_n_batches
        should_eval = ((step + 1) % eval_every_n_steps == 0) or (step + 1 == num_steps)
        if should_eval:
            print("Evaluation")

            # TODO: save best model to path
            # model.save_pretrained(args.checkpoint_dir)
            # tokenizer.save_pretrained(args.checkpoint_dir)

            # eval using vLLM
            if torch.cuda.is_available() and eval_model:
                load_policy_into_vllm_instance(model, eval_model)

                eval_res = evaluate_vllm(
                    eval_model,
                    r1_zero_reward_fn,
                    test_prompt_strs,
                    test_ground_truth_strs,
                    eval_sampling_params,
                )
                eval_reward_stat = eval_res.reward_stat
                print(eval_reward_stat)
                wandb_data["format_mean"] = eval_reward_stat["format"]["mean"]
                wandb_data["format_sum"] = eval_reward_stat["format"]["sum"]
                wandb_data["answer_mean"] = eval_reward_stat["answer"]["mean"]
                wandb_data["answer_sum"] = eval_reward_stat["answer"]["sum"]
                wandb_data["final_mean"] = eval_reward_stat["final"]["mean"]
                wandb_data["final_sum"] = eval_reward_stat["final"]["sum"]
            else:
                print("CUDA not available - skipping evaluation using vLLM")

        wandb_data["step"] = step
        wandb_data["lr"] = lr_scheduler.get_last_lr()[0]
        wandb_data["loss"] = loss.item()

        wandb.log(wandb_data)


if __name__ == "__main__":
    main()
