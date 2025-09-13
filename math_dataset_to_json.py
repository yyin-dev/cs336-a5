"""
Converts MATH dataset from https://github.com/sail-sg/understand-r1-zero into
JSONL fromat for convenience.

features:
    problem:
    solution:
    answer: the number from solution
    subject: algebra, etc.
    level: 1-5
    unique_id
    gold_solution_steps: divide 'solution' into steps

We split the math-12k dataset into train/test set manually.

$ uv run math_dataset_to_json.py --input ./data/math/train --train-set-output ./data/math/train.jsonl --test-set-output ./data/math/test.jsonl
"""

import torch
import json
from datasets import Dataset, load_from_disk
from argparse import ArgumentParser


def save(dataset: Dataset, filepath: str):
    with open(filepath, "w") as f:
        for item in dataset:
            json_obj = {"question": item["problem"], "answer": item["solution"]}
            f.write(json.dumps(json_obj) + "\n")

    print(f"Saved {len(dataset)} items to {filepath}")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input", type=str, help="File path of MATH 12k", required=True
    )
    parser.add_argument("--train-set-output", type=str, help="File path", required=True)
    parser.add_argument("--test-set-output", type=str, help="File path", required=True)
    args = parser.parse_args()

    dataset = load_from_disk(args.input)
    print(f"Dataset: {dataset}")

    res = dataset.train_test_split(test_size=0.1, train_size=0.9, seed=42)
    print(f"Train-test split res: {res}")

    save(res["train"], args.train_set_output)
    save(res["test"], args.test_set_output)


if __name__ == "__main__":
    main()
