import logging
import os
import random
import zipfile
from datetime import datetime
from pathlib import Path

import ray

from .trial_scheduler import TrialScheduler
from .trial_state import TrialResult, TrialState
from .utils import DataloaderFactory, TrainStepFunction, WorkerType
from .worker import generate_all_workers


def get_tuner_logger() -> logging.Logger:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(Path.cwd()) / "logs" / timestamp
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("Tuner")

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # 或者選擇更合適的級別

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s TUNER -- %(message)s",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)  # 只顯示 INFO 級別以上的訊息
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(Path(log_dir) / "tuner.log")
        file_handler.setLevel(logging.DEBUG)  # 記錄所有級別的日誌
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# NOTE:
# model 的建立的時間,
# batch_size 對於 throughput 計算
@ray.remote
class Tuner:
    def __init__(
        self,
        trial_states: list[TrialState],
        train_step: TrainStepFunction,
        dataloader_factory: DataloaderFactory,
    ) -> None:
        self.trial_states = trial_states
        self.logger = get_tuner_logger()

        self.logger.info("總共 %d 個 Trial", len(trial_states))
        self.logger.info("\n".join([str(t.hyperparameter) for t in trial_states]))

        self.workers = generate_all_workers(
            ray.get_runtime_context().current_actor,
            train_step=train_step,
            dataloader_factory=dataloader_factory,
        )

        self.scheduler: TrialScheduler = TrialScheduler(
            ray.get_runtime_context().current_actor,
            self.workers,
            trial_states,
        )
        self.trial_result: TrialResult = TrialResult()

        for trial in self.trial_states:
            self.trial_result.record_trial_progress(trial)

    def run(self) -> None:
        self.logger.info("開始訓練")
        self.scheduler.run()
        self.logger.info("結束訓練")
        self.scheduler.get_workers_logs()

    def update_trial_result(self, trial_state: TrialState) -> None:
        self.trial_result.update_trial_result(trial_state)
        self.logger.info(
            "History Best: %.6f %s",
            self.trial_result.history_best[0],
            self.trial_result.history_best[1],
        )

    def get_quantile_trial(
        self,
        ratio: float = 0.25,
    ) -> tuple[list[TrialState], list[TrialState]]:
        return self.trial_result.get_quantile(ratio)

    def record_trial_progress(self, trial_state: TrialState) -> None:
        self.trial_result.record_trial_progress(trial_state)
        self.trial_result.display_trial_progress()

    def mutation(self, trial_state: TrialState) -> TrialState:
        self.logger.info(
            "Trial %d: 執行mutation, 原始超參數: %s",
            trial_state.id,
            trial_state.hyperparameter,
        )

        lower_quantile, upper_quantile = self.get_quantile_trial()

        chose_trial = random.choice(upper_quantile)
        hyperparameter = chose_trial.hyperparameter
        hyperparameter.lr *= 0.8
        hyperparameter.momentum *= 1.2

        trial_state.hyperparameter = hyperparameter
        trial_state.checkpoint = chose_trial.checkpoint

        self.logger.info(
            "Trial-%d Iter-%d, 結束mutation, 新超參數: %s",
            trial_state.id,
            trial_state.iteration,
            trial_state.hyperparameter,
        )

        return trial_state

    def get_min_iteration_trial(self) -> tuple[int, int]:
        cpu_trial = [
            trial
            for trial in self.trial_result.trial_progress.values()
            if trial.worker_type == WorkerType.CPU
        ]
        target = min(cpu_trial, key=lambda x: x.iteration)
        return target.worker_id, target.id

    def get_trial_progress(self) -> list[TrialState]:
        return self.trial_result.get_trial_progress()

    def submit_trial(self, trial_state: TrialState) -> None:
        self.scheduler.submit_trial(trial_state)

    def get_zipped_log(self) -> bytes:
        log_dir = None

        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_dir = Path(handler.baseFilename).parent  # 取得資料夾路徑
                break
        if log_dir is None:
            msg = "log_dir not found."
            raise FileNotFoundError(msg)

        zip_path = "./logs.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(log_dir):
                for file in files:
                    abs_file = Path(root) / file
                    rel_path = os.path.relpath(abs_file, log_dir)
                    zf.write(abs_file, arcname=rel_path)

        with Path(zip_path).open("rb") as f:
            return f.read()

        return None
