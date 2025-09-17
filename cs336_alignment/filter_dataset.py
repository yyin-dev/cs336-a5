from datasets import Dataset, load_from_disk
from argparse import ArgumentParser
from drgrpo_grader import extract_answer, grade


def main():
    parser = ArgumentParser()
    parser.add_argument("--train-dataset", type=str, required=True)
    args = parser.parse_args()

    dataset = load_from_disk(args.train_dataset)

    # Dataset
    unfiltered = dataset.select_columns(["problem", "solution", "answer"])

    # Track indices that get filtered out
    correct_indices = []
    incorrect_indices = []

    for i in range(len(unfiltered)):
        example = unfiltered[i]
        extracted_answer = extract_answer(example["solution"])
        if extracted_answer is not None and grade(extracted_answer, example["answer"]):
            correct_indices.append(i)
        else:
            incorrect_indices.append(i)

    filtered = unfiltered.select(correct_indices)

    print(f"Unfiltered: {len(unfiltered)}")
    print(f"Filtered: {len(filtered)}")
    print(f"Filtered out {len(incorrect_indices)} examples")
    print(
        f"Filtered out indices: {incorrect_indices[:20]}..."
    )  # Show first 20 to avoid spam


if __name__ == "__main__":
    main()
