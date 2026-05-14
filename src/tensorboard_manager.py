from pathlib import Path

import ray
from torch.utils.tensorboard import SummaryWriter


@ray.remote
class TensorBoardManager:
    def __init__(self, base_log_dir: Path, num_trials: int, start_time: float) -> None:
        self.base_log_dir = base_log_dir
        self.global_writer: SummaryWriter | None = None
        self.trial_writers: dict[int, SummaryWriter] = {}
        self.num_trials = num_trials
        self.start_time = start_time

        self._initialize_writers()

    def _initialize_writers(self) -> None:
        global_log_dir = self.base_log_dir / "global"
        global_log_dir.mkdir(parents=True, exist_ok=True)
        self.global_writer = SummaryWriter(log_dir=str(global_log_dir))

        for tid in range(self.num_trials):
            trial_log_dir = self.base_log_dir / "trials" / f"trial_{tid}"
            trial_log_dir.mkdir(parents=True, exist_ok=True)
            self.trial_writers[tid] = SummaryWriter(log_dir=str(trial_log_dir))

    def get_trial_writer(self, trial_id: int) -> SummaryWriter:
        if trial_id not in self.trial_writers:
            msg = f"No writer found for trial_id: {trial_id}"
            raise ValueError(msg)
        return self.trial_writers[trial_id]

    def get_global_writer(self) -> SummaryWriter:
        if self.global_writer is None:
            msg = "Global writer not initialized."
            raise RuntimeError(msg)
        return self.global_writer

    def add_acc_to_trial(
        self,
        trial_id: int,
        scalar_value: float,
        generation: int,
    ) -> None:
        writer = self.get_trial_writer(trial_id)
        writer.add_scalar(
            "accuracy",
            scalar_value,
            generation,
        )

    def add_best_acc_to_global(
        self,
        scalar_value: float,
        timestamp: float,
    ) -> None:
        if self.global_writer:
            self.global_writer.add_scalar(
                "Best Accuracy",
                scalar_value,
                timestamp - self.start_time,
            )

    def close_all_writers(self) -> None:
        if self.global_writer:
            self.global_writer.close()
        for writer in self.trial_writers.values():
            writer.close()
