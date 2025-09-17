from vllm import LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers.modeling_utils import PreTrainedModel
from unittest.mock import patch
from datasets import Dataset, load_from_disk
import torch
import math

R1_ZERO_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""

R1_ZERO_OUTPUT = """ {solution} </think> <answer> {answer} </answer>"""


def fit_r1_zero_question(question: str):
    return R1_ZERO_PROMPT.replace("{question}", question)


def fit_r1_zero_output(solution: str, answer: str) -> str:
    return R1_ZERO_OUTPUT.replace("{solution}", solution).replace("{answer}", answer)


def ceiling_dev(x, y):
    return (x + y - 1) // y


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


def _cosine_lr_schedule_with_warmup(t, lr_max, lr_min, T_w, T_c):
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
        lr = _cosine_lr_schedule_with_warmup(
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


def load_math_train_using_r1_zero_prompt(path: str):
    """
    Returns: (prompt strs, output strs) formated in R1-zero format
    """
    train_dataset: Dataset = load_from_disk(path)  # type: ignore
    prompt_strs = [fit_r1_zero_question(v["problem"]) for v in train_dataset]
    output_strs = [
        fit_r1_zero_output(v["solution"], v["answer"]) for v in train_dataset
    ]

    return prompt_strs, output_strs


def load_math_test_using_r1_zero_prompt(path: str):
    test_dataset: Dataset = load_from_disk(path)  # type: ignore
    test_dataset = test_dataset.select_columns(["problem", "answer"])
    test_prompt_strs = [fit_r1_zero_question(qa["problem"]) for qa in test_dataset]
    test_ground_truth_strs = [qa["answer"] for qa in test_dataset]

    return test_prompt_strs, test_ground_truth_strs


def sample(train_input_ids, train_labels, train_response_mask, index, device):
    input_ids = torch.index_select(train_input_ids, 0, index).to(device)
    labels = torch.index_select(train_labels, 0, index).to(device)
    response_mask = torch.index_select(train_response_mask, 0, index).to(device)
    return (input_ids, labels, response_mask)
