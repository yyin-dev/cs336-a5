"""
Driver script for running expert iteration experiments with different parameter combinations.
"""

import subprocess
import sys

# Define parameter combinations to test
# Format: (expert_iteration_batch_size, rollout, epochs, batch_size, microbatch_size, lr)
PARAM_COMBINATIONS = [
    # (512, 2, 2),
    # (512, 2, 4),
    # (512, 2, 8),
    # (512, 4, 2),
    (512, 4, 4),
    # (512, 4, 8),
    # (1024, 2, 2),
    (1024, 2, 4),
    (1024, 2, 8),
    (2048, 1, 2),
    (2048, 1, 4),
    (2048, 1, 8),
]

# Fixed parameters
TRAIN_SET = "./data/math/train"
TEST_SET = "./data/math/test"


def run_experiment(
    expert_iteration_batch_size: int,
    rollout: int,
    epochs: int,
    batch_size: int = 8,
    microbatch_size: int = 2,
    lr: float = 5e-5,
    train_set: str = TRAIN_SET,
    test_set: str = TEST_SET,
) -> int:
    """Run a single expert iteration experiment with the given parameters."""

    cmd = [
        "uv",
        "run",
        "cs336_alignment/expert_iteration_experiment.py",
        "--expert-iteration-batch-size",
        str(expert_iteration_batch_size),
        "--rollout",
        str(rollout),
        "--batch-size",
        str(batch_size),
        "--microbatch-size",
        str(microbatch_size),
        "--train-set",
        train_set,
        "--test-set",
        test_set,
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
    ]

    print(f"Running experiment with params:")
    print(f"  Expert iteration batch size: {expert_iteration_batch_size}")
    print(f"  Rollout: {rollout}")
    print(f"  Batch size: {batch_size}")
    print(f"  Microbatch size: {microbatch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)

    try:
        result = subprocess.run(cmd, check=True)
        print(f"Experiment completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with return code {e.returncode}")
        return e.returncode


def main():
    """Run all parameter combinations."""
    print(f"Starting expert iteration experiment driver")
    print(f"Will run {len(PARAM_COMBINATIONS)} parameter combinations")
    print("=" * 60)

    failed_experiments = []

    for i, params in enumerate(PARAM_COMBINATIONS, 1):
        print(f"\nExperiment {i}/{len(PARAM_COMBINATIONS)}")

        returncode = run_experiment(*params)

        if returncode != 0:
            failed_experiments.append((i, params, returncode))

        print("=" * 60)

    # Summary
    print(f"\nSUMMARY:")
    print(f"Total experiments: {len(PARAM_COMBINATIONS)}")
    print(f"Successful: {len(PARAM_COMBINATIONS) - len(failed_experiments)}")
    print(f"Failed: {len(failed_experiments)}")

    if failed_experiments:
        print(f"\nFailed experiments:")
        for exp_num, params, returncode in failed_experiments:
            print(f"  Experiment {exp_num}: {params} (exit code: {returncode})")
        sys.exit(1)
    else:
        print(f"All experiments completed successfully!")


if __name__ == "__main__":
    main()
