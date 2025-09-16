"""
uv run cs336_alignment/sft_experiment.py \
    --batch-size 8 \
    --unique-examples 128 \
    --train-set ./data/math/train \
    --test-set ./data/math/test \
    --gradient-accumulation-steps 4 \
    --lr 1e-5 \
    --checkpoint-dir ../checkpoints
"""

from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
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
import math
from sft import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
)
from math_baseline import evaluate_vllm
from drgrpo_grader import r1_zero_reward_fn

QWEN = "Qwen/Qwen2.5-Math-1.5B"
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

R1_ZERO_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""

R1_ZERO_OUTPUT = """ {solution} </think> <answer> {answer} </answer>"""


def fit_prompt(prompt_template: str, question: str) -> str:
    return prompt_template.replace("{question}", question)


def fit_output(output_template: str, solution: str, answer: str) -> str:
    return output_template.replace("{solution}", solution).replace("{answer}", answer)


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


def ceiling_dev(x, y):
    return (x + y - 1) // y


def sample(train_input_ids, train_labels, train_response_mask, index, device):
    input_ids = torch.index_select(train_input_ids, 0, index).to(device)
    labels = torch.index_select(train_labels, 0, index).to(device)
    response_mask = torch.index_select(train_response_mask, 0, index).to(device)
    return (input_ids, labels, response_mask)


def main():
    parser = ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--unique-examples", type=int, required=True)
    parser.add_argument("--train-set", type=str, required=True)
    parser.add_argument("--test-set", type=str, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--init-eval", action="store_true")
    args = parser.parse_args()

    # wandb
    run = wandb.init(
        name=f"sft-{args.unique_examples}-ep{args.epochs}-lr{args.lr}",
        config={
            "unique_examples": args.unique_examples,
            "epochs": args.epochs,
            "lr": args.lr,
            "gradient_accum_steps": args.gradient_accumulation_steps,
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
    train_prompt_strs = [
        fit_prompt(R1_ZERO_PROMPT, qa["problem"]) for qa in train_dataset
    ]
    train_output_strs = [
        fit_output(R1_ZERO_OUTPUT, qa["solution"], qa["answer"]) for qa in train_dataset
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
    test_prompt_strs = [
        fit_prompt(R1_ZERO_PROMPT, qa["problem"]) for qa in test_dataset
    ]
    test_ground_truth_strs = [qa["answer"] for qa in test_dataset]
    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop="</answer>",
        include_stop_str_in_output=True,
    )

    microbatch_size = max(args.batch_size // args.gradient_accumulation_steps, 1)
    print(f"Microbatch size: {microbatch_size}")

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
    num_optimizer_steps = ceiling_dev(args.unique_examples, args.batch_size) * args.epochs

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

        print(input_ids.shape, labels.shape, response_mask.shape)

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
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            normalize_constant=1.0,
        )

        print(f"Loss: {loss.item()}, stat: {stat}")

        clip_grad_norm_(model.parameters(), max_norm=1.0)

        if (step + 1) % args.gradient_accumulation_steps == 0:
            opt.step()
            opt.zero_grad()
            lr_scheduler.step()

        # evaluate every few batches using vLLM
        save_every_n_batches = 5
        if (step + 1) % (args.gradient_accumulation_steps * save_every_n_batches) == 0:
            print("Evaluation")

            # save model to path
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
