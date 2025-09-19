"""
Driver script for running GRPO experiments with different parameter combinations.
"""

import subprocess
import sys
from dataclasses import dataclass
from typing import List


@dataclass
class GRPOExperimentConfig:
    """Configuration for a GRPO experiment run."""

    # Core GRPO parameters
    n_grpo_steps: int = 200
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 256
    group_size: int = 8
    sampling_temperature: float = 1.0
    epochs_per_rollout_batch: int = 1
    train_batch_size: int = 256
    train_microbatch_size: int = 2
    loss_type: str = "reinforce_with_baseline"
    use_std_normalization: bool = True
    cliprange: float = 0.2
    eval_every_n_steps: int = 5

    # Dataset parameters
    train_dataset: str = "./data/math/train"
    test_dataset: str = "./data/math/test"

    # Technical parameters
    sampling_min_tokens: int = 4
    sampling_max_tokens: int = 1024
    gradient_accumulation_steps: int = 128
    gpu_memory_utilization: float = 0.85

    def to_command_args(self) -> List[str]:
        """Convert config to command line arguments."""
        return [
            "--train-dataset", self.train_dataset,
            "--test-dataset", self.test_dataset,
            "--n-grpo-steps", str(self.n_grpo_steps),
            "--learning-rate", str(self.learning_rate),
            "--advantage-eps", str(self.advantage_eps),
            "--rollout-batch-size", str(self.rollout_batch_size),
            "--group-size", str(self.group_size),
            "--sampling-temperature", str(self.sampling_temperature),
            "--sampling-min-tokens", str(self.sampling_min_tokens),
            "--sampling-max-tokens", str(self.sampling_max_tokens),
            "--epochs-per-rollout-batch", str(self.epochs_per_rollout_batch),
            "--train-batch-size", str(self.train_batch_size),
            "--train-microbatch-size", str(self.train_microbatch_size),
            "--gradient-accumulation-steps", str(self.gradient_accumulation_steps),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--loss-type", self.loss_type,
            "--use-std-normalization" if self.use_std_normalization else "--no-use-std-normalization",
            "--cliprange", str(self.cliprange),
            "--eval-every-n-steps", str(self.eval_every_n_steps),
        ]

    def description(self) -> str:
        """Get a human-readable description of this config."""
        return f"{self.loss_type}_rollout{self.rollout_batch_size}_grp{self.group_size}_lr{self.learning_rate}_ep{self.epochs_per_rollout_batch}"


# Define parameter combinations to test
EXPERIMENT_CONFIGS = [
    # On-policy REINFORCE with baseline experiments
    GRPOExperimentConfig(
        rollout_batch_size=256,
        group_size=8,
        learning_rate=1e-5,
        epochs_per_rollout_batch=1,
        loss_type="reinforce_with_baseline",
    ),
    GRPOExperimentConfig(
        rollout_batch_size=256,
        group_size=16,
        learning_rate=1e-5,
        epochs_per_rollout_batch=1,
        loss_type="reinforce_with_baseline",
    ),

    # Off-policy GRPO-Clip experiments
    GRPOExperimentConfig(
        rollout_batch_size=256,
        group_size=8,
        learning_rate=1e-5,
        epochs_per_rollout_batch=3,
        loss_type="grpo_clip",
        cliprange=0.2,
    ),
    GRPOExperimentConfig(
        rollout_batch_size=256,
        group_size=16,
        learning_rate=1e-5,
        epochs_per_rollout_batch=3,
        loss_type="grpo_clip",
        cliprange=0.2,
    ),

    # Learning rate ablations
    GRPOExperimentConfig(
        rollout_batch_size=256,
        group_size=8,
        learning_rate=5e-6,
        epochs_per_rollout_batch=1,
        loss_type="reinforce_with_baseline",
    ),
    GRPOExperimentConfig(
        rollout_batch_size=256,
        group_size=8,
        learning_rate=2e-5,
        epochs_per_rollout_batch=1,
        loss_type="reinforce_with_baseline",
    ),
]


def run_experiment(config: GRPOExperimentConfig) -> int:
    """Run a single GRPO experiment with the given configuration."""

    cmd = [
        "uv",
        "run",
        "cs336_alignment/grpo_experiment.py",
    ] + config.to_command_args()

    print(f"Running GRPO experiment: {config.description()}")
    print(f"Configuration:")
    print(f"  Loss type: {config.loss_type}")
    print(f"  Rollout batch size: {config.rollout_batch_size}")
    print(f"  Group size: {config.group_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs per rollout: {config.epochs_per_rollout_batch}")
    print(f"  Microbatch size: {config.train_microbatch_size}")
    if config.loss_type == "grpo_clip":
        print(f"  Clip range: {config.cliprange}")
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
    print(f"Starting GRPO experiment driver")
    print(f"Will run {len(EXPERIMENT_CONFIGS)} parameter combinations")
    print("=" * 60)

    failed_experiments = []

    for i, config in enumerate(EXPERIMENT_CONFIGS, 1):
        print(f"\nExperiment {i}/{len(EXPERIMENT_CONFIGS)}")

        returncode = run_experiment(config)

        if returncode != 0:
            failed_experiments.append((i, config, returncode))

        print("=" * 60)

    # Summary
    print(f"\nSUMMARY:")
    print(f"Total experiments: {len(EXPERIMENT_CONFIGS)}")
    print(f"Successful: {len(EXPERIMENT_CONFIGS) - len(failed_experiments)}")
    print(f"Failed: {len(failed_experiments)}")

    if failed_experiments:
        print(f"\nFailed experiments:")
        for exp_num, config, returncode in failed_experiments:
            print(f"  Experiment {exp_num}: {config.description()} (exit code: {returncode})")
        sys.exit(1)
    else:
        print(f"All experiments completed successfully!")


if __name__ == "__main__":
    main()