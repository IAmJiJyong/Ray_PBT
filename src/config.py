import json
from pathlib import Path

config = {}
path = Path.resolve(Path(Path(__file__).parent) / "../config.json")

if Path.exists(path):
    with Path(path).open("r") as f:
        config = json.load(f)

DATASET_PATH: str = config.get("dataset_path", "~/Documents/dataset/")
STOP_ACCURACY: float = config.get("stop_accuracy", 0.9)
MAX_GENERATION: int = config.get("max_generation", 5)
ITERATION_PER_GENERATION: int = config.get("iteration_per_generation", 20)
TRIAL_PROGRESS_OUTPUT_PATH: Path = Path(
    config.get(
        "trial_progress_output_path",
        "~/Documents/Ray_PBT/trial_progress.output",
    ),
).expanduser()
GPU_TRIALS_LIMIT: int = config.get("gpu_trials_limit", 3)
CPU_TRIALS_LIMIT: int = config.get("cpu_trials_limit", 1)
MAX_BATCH_SIZE: int = config.get("max_batch_size", 512)
MIN_BATCH_SIZE: int = config.get("min_batch_size", 32)
PERTURBATION_FACTOR: int = config.get("perturbation_factor", 0.2)
TUNE_TUPE: str = config.get("tune_type", "pbt")
TEST_NUM: int = config.get("test_num", 1)
