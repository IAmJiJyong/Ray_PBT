import logging
import os
import time
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import ray

from .trial_manager import NESTrialManager, PBTTrialManager
from .trial_scheduler import TrialScheduler
from .trial_state import PartialTrialState, TrialState
from .utils import (
    DataloaderFactory,
    TrainStepFunction,
    TrialStatus,
    WorkerType,
    get_head_node_address,
)
from .utils_nes import TuneType
from .worker_manager import WorkerManager

if TYPE_CHECKING:
    from ray.actor import ActorHandle


def get_tuner_logger() -> logging.Logger:
    timestamp = (datetime.now(UTC) + timedelta(hours=8)).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path.cwd() / "logs" / timestamp
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
        tune_type: TuneType = TuneType.PBT,
    ) -> None:
        self.logger = get_tuner_logger()
        self.logger.info("總共 %d 個 Trial", len(trial_states))
        self.tune_type = tune_type

        match self.tune_type:
            case TuneType.PBT:
                self.trial_manager: ActorHandle = PBTTrialManager.options(
                    max_concurrency=10,
                    num_cpus=1,
                    resources={f"node:{get_head_node_address()}": 0.01},
                ).remote(trial_states)  # type: ignore[reportGeneralTypeIssues]
            case TuneType.NES:
                self.trial_manager: ActorHandle = NESTrialManager.options(
                    max_concurrency=10,
                    num_cpus=1,
                    resources={f"node:{get_head_node_address()}": 0.01},
                ).remote(trial_states)  # type: ignore[reportGeneralTypeIssues]

        self.worker_manager = WorkerManager(
            ray.get_runtime_context().current_actor,
            self.trial_manager,
            train_step=train_step,
            dataloader_factory=dataloader_factory,
        )

        self.scheduler: TrialScheduler = TrialScheduler(
            self.worker_manager,
            self.trial_manager,
        )

    def run(self) -> None:
        start = time.perf_counter()
        self.logger.info("開始訓練")
        self.scheduler.run()
        self.logger.info("結束訓練")
        end = time.perf_counter()
        run_time = end - start
        self.logger.info("訓練總時長: %.2f 秒", run_time)
        self.scheduler.get_workers_logs()
        self.logger.info("Assign: %d", self.worker_manager.assign_count["assign"])
        self.logger.info("Locality: %d", self.worker_manager.assign_count["locality"])
        acc = ray.get(self.trial_manager.get_history_best_result.remote()).accuracy  # type: ignore[reportGeneralTypeIssues]
        self.record_results(
            time=run_time,
            acc=acc,
        )

    def record_results(self, time: float, acc: float) -> None:
        new_row = {"Time": time, "Accuracy": acc}
        log_dir = Path("~/Documents/workspace/shaogu/Ray_PBT/log").expanduser()
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{self.tune_type}_results.csv"
        if log_file.exists():
            df = pd.read_csv(log_file)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])
        df.to_csv(log_file, index=False)

    def on_trial_complete(
        self,
        worker_id: int,
        trial_id: int,
        worker_type: WorkerType,
        partial: PartialTrialState,
    ) -> None:
        if "accuracy" not in partial:
            self.logger.warning(
                "Worker %d 回傳的 Trial %d 沒有 accuracy",
                worker_id,
                trial_id,
            )
            msg = "Worker %d 回傳的 Trial %d Partial沒有 accuracy"
            raise ValueError(msg)

        self.logger.info(
            "✅ Worker %d Trial %d 完成, Accuracy: %.2f",
            worker_id,
            trial_id,
            partial["accuracy"],
        )

        partial["worker_id"] = -1
        partial["worker_type"] = None
        ray.get(
            self.trial_manager.transition_status.remote(
                trial_id,
                TrialStatus.TERMINATED,
                partial,
            ),  # type: ignore[reportGeneralTypeIssues]
        )
        self.worker_manager.release_slots(worker_id, trial_id)

        if ray.get(self.trial_manager.is_finish.remote()):  # type: ignore[reportGeneralTypeIssues]
            self.scheduler.finish()
            return

        self.scheduler.assign_trial_to_worker(
            worker_id,
            worker_type,
        )

    def on_trial_step_complete(
        self,
        worker_id: int,
        trial_id: int,
        worker_type: WorkerType,
        partial: PartialTrialState,
    ) -> None:
        if "accuracy" not in partial or "generation" not in partial:
            self.logger.warning(
                "Worker %d 回傳的 Trial %d 沒有 accuracy 或 generation",
                worker_id,
                trial_id,
            )
            error_msg = "Worker %d 回傳的 Trial %d Partial沒有 accuracy 或 generation"
            raise ValueError(error_msg)

        self.logger.info(
            "🔃 Worker %d 回傳未完成 Trial %d, Iteration: %d, Accuracy: %.2f",
            worker_id,
            trial_id,
            partial["generation"],
            partial["accuracy"],
        )

        partial["worker_id"] = -1
        partial["worker_type"] = None
        ray.get(
            self.trial_manager.transition_status.remote(  # type: ignore[reportGeneralTypeIssues]
                trial_id,
                TrialStatus.PENDING,
                partial,
            ),
        )

        self.worker_manager.release_slots(worker_id, trial_id)

        self.scheduler.assign_trial_to_worker(
            worker_id,
            worker_type,
        )

    def on_trial_need_mutation(
        self,
        worker_id: int,
        trial_id: int,
        worker_type: WorkerType,
        partial: PartialTrialState,
    ) -> None:
        self.logger.info(
            "🔃 Worker %d 回傳 Trial %d 執行 mutation",
            worker_id,
            trial_id,
        )

        self.logger.info("Trial %d: 執行mutation", trial_id)
        mutation_partial = ray.get(self.trial_manager.mutation.remote())  # type: ignore[reportGeneralTypeIssues]

        self.logger.info(
            "Trial %d 結束mutation, 新超參數: %s",
            trial_id,
            mutation_partial["hyperparameter"],
        )

        partial["worker_id"] = -1
        partial["worker_type"] = None
        ray.get(
            self.trial_manager.transition_status.remote(  # type: ignore[reportGeneralTypeIssues]
                trial_id,
                TrialStatus.PENDING,
                partial | mutation_partial,
            ),
        )

        self.worker_manager.release_slots(worker_id, trial_id)

        self.scheduler.assign_trial_to_worker(
            worker_id,
            worker_type,
        )

    def get_zipped_log(self) -> bytes:
        log_dir = None

        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_dir = Path(handler.baseFilename).parent  # 取得資料夾路徑
                break

        if log_dir is None:
            msg = "log_dir not found."
            raise FileNotFoundError(msg)

        trial_manager_log_content = ray.get(
            self.trial_manager.get_log_file.remote(),  # type: ignore[reportGeneralTypeIssues]
        )
        with (log_dir / "trial_manager.log").open("w") as f:
            f.write(trial_manager_log_content)

        worker_manager_log_content = self.worker_manager.get_log_file()

        with (log_dir / "worker_manager.log").open("w") as f:
            f.write(worker_manager_log_content)

        zip_path = "./logs.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(log_dir):
                for file in files:
                    abs_file = Path(root) / file
                    rel_path = os.path.relpath(abs_file, log_dir)
                    zf.write(abs_file, arcname=rel_path)

        with Path(zip_path).open("rb") as f:
            return f.read()
