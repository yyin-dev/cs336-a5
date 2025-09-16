import json
from vllm import LLM, SamplingParams
from drgrpo_grader import r1_zero_reward_fn
from argparse import ArgumentParser
import statistics
from typing import Callable, List
from dataclasses import dataclass
from datasets import load_from_disk

"""
Train
{'problem': 'How many units long is a segment whose endpoints are $(-4,1)$ and $(1,13)$?', 
 'solution': 'We use the distance formula: $\\sqrt{(-4 - 1)^2 + (1 - 13)^2},$ which is $\\sqrt{25 + 144} = \\sqrt{169} = \\boxed{13}$.\n\n- OR -\n\nWe note that the points $(-4,1)$, $(1,13)$, and $(1,1)$ form a right triangle with legs of length 5 and 12. $(5,12,13)$ is a Pythagorean triple, so the hypotenuse has length $\\boxed{13}$.', 
 'answer': '13', 
 'subject': 'Algebra', 
 'level': 2, 
 'unique_id': 'test/algebra/1570.json', 
 'gold_solution_steps': ['We use the distance formula:', '$\\sqrt{(-4 - 1)^2 + (1 - 13)^2},$ which is $\\sqrt{25 + 144} = \\sqrt{169} = \\boxed{13}$.', '- OR - We note that the points $(-4,1)$, $(1,13)$,', 'and $(1,1)$ form a right triangle with legs of length 5 and 12. $(5,12,13)$ is a Pythagorean triple, so the hypotenuse has length $\\boxed{13}$.']
}

Test:
{'problem': 'Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$', 
 'answer': '\\left( 3, \\frac{\\pi}{2} \\right)', 
 'difficulty': 2.0
}
"""


@dataclass
class EvalResult:
    responses: List[str]
    rewards: List[dict[str, float]]
    reward_stat: dict[str, dict[str, float]]


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    groud_truths: List[str],
    eval_sampling_params: SamplingParams,
) -> EvalResult:
    responses = vllm_model.generate(prompts, eval_sampling_params)
    responses = [output.outputs[0].text for output in responses]
    rewards = [
        reward_fn(response, truth) for response, truth in zip(responses, groud_truths)
    ]

    # Print some metrics
    format_rewards = [reward["format_reward"] for reward in rewards]
    answer_rewards = [reward["answer_reward"] for reward in rewards]
    final_rewards = [reward["reward"] for reward in rewards]

    def print_reward_stat(rewards, name):
        print(
            f"{name}: mean {statistics.mean(rewards)}, median {statistics.median(rewards)}, sum {sum(rewards)}, min {min(rewards)}, max {max(rewards)}"
        )

    reward_stat = {
        "format": {
            "mean": statistics.mean(format_rewards),
            "sum": sum(format_rewards),
        },
        "answer": {
            "mean": statistics.mean(answer_rewards),
            "sum": sum(answer_rewards),
        },
        "final": {
            "mean": statistics.mean(final_rewards),
            "sum": sum(final_rewards),
        },
    }

    print_reward_stat(format_rewards, "format rewards")
    print_reward_stat(answer_rewards, "answer rewards")
    print_reward_stat(final_rewards, "final rewards")

    return EvalResult(responses, rewards, reward_stat)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The prompt template should contain a '{required}.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Generate at most this number of responses",
    )
    parser.add_argument(
        "--result", type=str, default="math_baseline.jsonl", help="Output path"
    )
    args = parser.parse_args()

    # load dataset
    dataset = load_from_disk(args.dataset)
    qas = [(qa["problem"], qa["answer"]) for qa in dataset]

    if args.limit is not None:
        qas = qas[: args.limit]

    # generate prompts
    with open(args.prompt, "r") as prompt_f:
        prompt_template = prompt_f.read()

    prompts = [fit_prompt(prompt_template, q) for (q, _) in qas]
    ground_truths = [a for (_, a) in qas]

    # evaluate LLM
    llm = LLM("Qwen/Qwen2.5-Math-1.5B")

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop="</answer>",
        include_stop_str_in_output=True,
    )

    # Setting fast=True/False doesn't affect format_reward, but does affect
    # answer_reward. For MATH baseline, fast=False almost 2x answer_reward.
    reward_fn = lambda x, y: r1_zero_reward_fn(x, y)
    result = evaluate_vllm(llm, reward_fn, prompts, ground_truths, sampling_params)

    # Save to file for analysis
    with open(args.result, "w") as output:
        for (question, truth), response, reward in zip(
            qas, result.responses, result.rewards
        ):
            datapoint = {
                "question": question,
                "truth": truth,
                "response": response,
                "reward": reward,
            }
            json.dump(datapoint, output)
            output.write("\n")  # Write a newline to produce jsonl

    print(f"Saved to {args.result}!")


def fit_prompt(prompt_template: str, question: str) -> str:
    return prompt_template.replace("{question}", question)


if __name__ == "__main__":
    main()
