import argparse
from datetime import UTC, datetime, timedelta
from itertools import islice
from pathlib import Path

import ray
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.config import (
    ITERATION_PER_GENERATION,
    MAX_GENERATION,
    STOP_ACCURACY,
)
from src.hyperparameter import BertHyperparameter
from src.task_strategy import BertSST2Task
from src.trial_state import TrialState
from src.tuner import Tuner
from src.utils import get_head_node_address, unzip_file

DEFAULT_DEVICE = torch.device("cpu")


def generate_trial_states(
    n: int = 1,
) -> list[TrialState]:
    return [
        TrialState(
            i,
            BertHyperparameter.random(),
        )
        for i in range(n)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_num", type=int, default=40)
    args = parser.parse_args()

    trial_num = args.trial_num

    print(f"STOP_ACCURACY {STOP_ACCURACY}")
    print(f"MAX_GENERATION {MAX_GENERATION}")
    print(f"ITERATION_PER_GENERATION {ITERATION_PER_GENERATION}")

    ray.init(
        runtime_env={
            "working_dir": ".",
            "excludes": [
                ".git",
                "test",
                "logs/*",
                "LICENSE",
                "README.md",
                ".venv",
                ".ruff_cache",
            ],
        },
    )
    trial_states = generate_trial_states(trial_num)

    bs_list = [32, 64, 128]
    for i in range(len(trial_states)):
        batch_size = bs_list[i % len(bs_list)]
        trial_states[i].hyperparameter.batch_size = batch_size

    tuner = Tuner.options(  # type: ignore[call-arg]
        max_concurrency=5,
        num_cpus=1,
        resources={f"node:{get_head_node_address()}": 0.01},
    ).remote(trial_states, BertSST2Task())

    ray.get(tuner.run.remote())  # type: ignore[call-arg]

    zip_logs_bytes: bytes = ray.get(tuner.get_zipped_log.remote())  # type: ignore[call-arg]

    time_stamp = (datetime.now(UTC) + timedelta(hours=8)).strftime("%Y-%m-%d_%H-%M-%S")
    zip_output_dir = Path("./logs") / "BERT" / f"Trial{trial_num}" / time_stamp

    zip_output_dir.mkdir(parents=True, exist_ok=True)
    zip_output_path = Path(zip_output_dir) / "logs.zip"
    with Path(zip_output_path).open("wb") as f:
        f.write(zip_logs_bytes)

    unzip_file(zip_output_path, zip_output_dir)  # type: ignore[call-arg]

    ray.shutdown()
