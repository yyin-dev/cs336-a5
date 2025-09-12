"""
Assumes that each line .jsonl is a json object:
- question
- truth
- response
- reward
  - format_reward
  - answer_reward
  - reward
"""

import json
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", required=True, type=str, help="JSONL file")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = [json.loads(line) for line in f]

    rewards = [d["reward"] for d in data]

    correct_with_both_format_and_answer_reward_1 = 0
    format_reward_1_answer_reward_0 = 0
    format_reward_0_answer_reward_0 = 0

    for reward in rewards:
        format_reward = reward["format_reward"]
        answer_reward = reward["answer_reward"]
        final_reward = reward["reward"]

        if format_reward == 1 and answer_reward == 1 and final_reward == 1:
            correct_with_both_format_and_answer_reward_1 += 1
        elif format_reward == 1 and answer_reward == 0:
            format_reward_1_answer_reward_0 += 1
        elif format_reward == 0 and answer_reward == 0:
            format_reward_0_answer_reward_0 += 1

    print(f"correct with both reward 1: {correct_with_both_format_and_answer_reward_1}")
    print(f"format reward 1, answer reward 0: {format_reward_1_answer_reward_0}")
    print(f"format reward 0, answer reward 0: {format_reward_0_answer_reward_0}")


if __name__ == "__main__":
    main()
