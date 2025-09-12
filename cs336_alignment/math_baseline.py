import json
from vllm import LLM, SamplingParams
from drgrpo_grader import r1_zero_reward_fn
from argparse import ArgumentParser
import statistics


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
    args = parser.parse_args()

    # load dataset
    qas = []
    with open(args.dataset, "r") as f:
        for line in f:
            qa = json.loads(line)
            q = qa["question"]
            a = qa["answer"]
            qas.append((q, a))

    if args.limit is not None:
        qas = qas[: args.limit]

    # generate prompts
    with open(args.prompt, "r") as prompt_f:
        prompt_template = prompt_f.read()

    prompts = [fit_prompt(prompt_template, q) for (q, _) in qas]

    # generate responses using vLLM
    llm = LLM("Qwen/Qwen2.5-Math-1.5B")

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop="</answer>",
        include_stop_str_in_output=True,
    )

    responses = llm.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in responses]

    # Compute reward
    rewards = [
        r1_zero_reward_fn(response, truth, fast=False)
        for (_, truth), response in zip(qas, responses)
    ]

    # Print some metrics
    format_rewards = [reward["format_reward"] for reward in rewards]
    answer_rewards = [reward["answer_reward"] for reward in rewards]
    final_rewards = [reward["reward"] for reward in rewards]

    def print_reward_stat(rewards, name):
        print(
            f"{name}: mean {statistics.mean(rewards)}, median {statistics.median(rewards)}, sum {sum(rewards)}, min {min(rewards)}, max {max(rewards)}"
        )

    print_reward_stat(format_rewards, "format rewards")
    print_reward_stat(answer_rewards, "answer rewards")
    print_reward_stat(final_rewards, "final rewards")

    # Save to file for analysis
    with open("math_baseline.jsonl", "w") as output:
        for (question, truth), response, reward in zip(qas, responses, rewards):
            datapoint = {
                "question": question,
                "truth": truth,
                "response": response,
                "reward": reward,
            }
            json.dump(datapoint, output)
            output.write("\n")  # Write a newline to produce jsonl

    print("Saved to math_baseline.jsonl!")


def fit_prompt(prompt_template: str, question: str) -> str:
    return prompt_template.replace("{question}", question)


if __name__ == "__main__":
    main()
