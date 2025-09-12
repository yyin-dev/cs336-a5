import os

# Configure vLLM download directory
os.environ["HF_HOME"] = "~/Desktop/cs336/Assignment5/hf_download"

import json
from vllm import LLM, SamplingParams
from drgrpo_grader import r1_zero_reward_fn
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The prompt template should contain a '{required}.",
    )
    args = parser.parse_args()

    # load dataset
    qas = []
    with open(args.dataset) as f:
        for line in f:
            qa = json.loads(line)
            q = qa["question"]
            a = qa["answer"]
            qas.append((q, a))

    # generate prompts
    with open(args.prompt) as prompt_f:
        prompt_template = prompt_f.read()

    prompts = [fit_prompt(prompt_template, q) for (q, _) in qas]
    prompts = prompts[:1]

    # generate outputs using vLLM
    llm = LLM("Qwen/Qwen2.5-Math-1.5B")

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop="</answer>",
        include_stop_str_in_output=True,
    )

    outputs = llm.generate(prompts, sampling_params)
    print(outputs)


def fit_prompt(prompt_template: str, question: str) -> str:
    return prompt_template.replace("{question}", question)


if __name__ == "__main__":
    main()
