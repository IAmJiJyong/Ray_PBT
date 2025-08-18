import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

import ray
from ray import ObjectRef
from ray.actor import ActorHandle

from .utils import WorkerType
from .worker_manager import WorkerManager

if TYPE_CHECKING:
    from .trial_state import TrialState


def gpu_scheduling(
    trial_manager: ActorHandle,
    worker_manager: WorkerManager,
    worker_id: int,
    logger: logging.Logger,
) -> None:
    # 若沒有任何 pending 的 Trial,結束
    if not ray.get(trial_manager.has_pending_trials.remote()):  # type: ignore[reportGeneralTypeIssues]
        logger.info("沒有待分配的 Trial")
        return

    selected_worker_entry = worker_manager.workers[worker_id]

    # 取得 iteration 次數最少的一組 pending trials
    trial_states: list[TrialState] = ray.get(
        trial_manager.get_pending_trials_with_min_iteration.remote(),  # type: ignore[reportGeneralTypeIssues]
    )

    # 優先挑選之前曾經在該 worker 上有 checkpoint 的 trial
    selected_trial = next(
        (
            t
            for t in trial_states
            if not t.last_checkpoint_location.is_empty()
            and t.last_checkpoint_location.worker_id == selected_worker_entry.id
        ),
        trial_states[0],  # 若無符合者, 則選第一個
    )

    # 設定 chunk_size、worker_id、worker_type 並標記為執行狀態
    selected_trial.set_chunk_size(
        ray.get(trial_manager.get_chunk_size.remote(selected_trial.generation)),  # type: ignore[reportGeneralTypeIssues]
    )

    trial_manager.transition_to_waiting.remote(selected_trial.id)
    trial_manager.update_trial.remote(
        selected_trial.id,
        {"worker_id": selected_worker_entry.id, "worker_type": WorkerType.GPU},
    )

    worker_manager.assign_trial_to_worker(
        WorkerType.GPU,
        selected_worker_entry.id,
        selected_trial,
    )


def cpu_scheduling(
    trial_manager: ActorHandle,
    worker_manager: WorkerManager,
    worker_id: int,
    logger: logging.Logger,
) -> None:
    # 若沒有任何 pending 的 Trial,結束
    if not ray.get(trial_manager.has_pending_trials.remote()):  # type: ignore[reportGeneralTypeIssues]
        logger.info("沒有待分配的 Trial")
        return

    # 選擇第一個可用的 worker
    selected_worker_entry = worker_manager.workers[worker_id]
    selected_trial = ray.get(
        trial_manager.get_nlargest_iteration_trials.remote(  # type: ignore[reportGeneralTypeIssues]
            len(worker_manager.workers),
        ),
    )

    if not selected_trial:
        return

    selected_trial = selected_trial[-1]
    # 設定 chunk_size(暫定為 2), 標記執行資訊
    selected_trial.set_chunk_size(2)
    selected_trial.worker_id = selected_worker_entry.id
    selected_trial.worker_type = WorkerType.CPU

    # 更新 Trial 狀態至 running
    trial_manager.transition_to_waiting.remote(selected_trial.id)
    trial_manager.update_trial.remote(
        {"worker_id": worker_id, "worker_type": WorkerType.CPU},
    )

    worker_manager.assign_trial_to_worker(
        WorkerType.CPU,
        selected_worker_entry.id,
        selected_trial,
    )


def stealing_strategy(
    worker_manager: WorkerManager,
    logger: logging.Logger,
) -> None:
    logger.info("嘗試從 CPU Worker 偷取任務")
    running_workers = (
        worker_entry
        for worker_entry in worker_manager.cpu_workers.values()
        if worker_entry.available_slots == 0
    )

    worker = next(running_workers, None)
    if worker is None:
        logger.info("沒有可用的 CPU Worker 來偷取任務")
        return

    logger.info("嘗試從 CPU Worker %d 偷取任務", worker.id)
    worker.ref.stealing_trial.remote(worker.active_trials[0])  # type: ignore[reportGeneralTypeIssues])


def get_trial_scheduler_logger() -> logging.Logger:
    """
    設置並返回一個日誌記錄器, 用於跟踪訓練過程中的 TrialScheduler 記錄。

    日誌將記錄到一個帶有時間戳的目錄中, 並包括在終端顯示和日誌文件中的訊息。

    Returns:
        logging.Logger: 配置好的 TrialScheduler 記錄器。
    """
    timestamp = (datetime.now(UTC) + timedelta(hours=8)).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path.cwd() / "logs" / timestamp
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("trial_scheduler")

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # 或者選擇更合適的級別

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s TRIAL_SCHEDULER -- %(message)s",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)  # 只顯示 INFO 級別以上的訊息
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(Path(log_dir) / "trial_scheduler.log")
        file_handler.setLevel(logging.DEBUG)  # 記錄所有級別的日誌
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class TrialScheduler:
    """
    試驗調度器, 負責管理和分配訓練試驗給可用的工作者。

    Attributes:
        trial_states (List[TrialState]): 當前待分配的試驗狀態列表。
        running_futures (List[ObjectRef]): 當前正在運行的訓練任務。
        completed_trial_state (List[TrialState]): 完成的試驗狀態列表。
        logger (logging.Logger): 記錄訓練過程的日誌記錄器。
        train_result (TrainResult): 用於記錄每個訓練結果的實例。
        workers (List[ActorHandle]): 可用的工作者列表。
    """

    def __init__(
        self,
        worker_manager: WorkerManager,
        trial_manager: ActorHandle,
    ) -> None:
        """
        初始化 TrialScheduler, 設置試驗狀態和工作者。

        Args:
            train_step (TrainStepFunction): 訓練步驟函數。
            trial_states (List[TrialState]): 初始的試驗狀態列表。
        """
        self.trial_manager = trial_manager
        self.worker_manager = worker_manager

        self.running_futures: list[ObjectRef] = []
        self.logger: logging.Logger = get_trial_scheduler_logger()

        self._previous_time: float = time.time()
        self.logger.info("初始化完成")
        self.finish_event = Event()

    def init_workers(self) -> None:
        for worker_entry in self.worker_manager.workers.values():
            num_cpus, num_gpus = (
                worker_entry.state.max_cpu_tasks,
                worker_entry.state.max_gpu_tasks,
            )

            self.logger.info(
                "Worker %d, num_cpus: %d num_gpus: %d",
                worker_entry.id,
                num_cpus,
                num_gpus,
            )

            trials: list[TrialState] = ray.get(
                self.trial_manager.acquire_pending_trials.remote(
                    worker_id=worker_entry.id,
                    n=num_cpus + num_gpus,
                ),  # type: ignore[reportGeneralTypeIssues]
            )

            worker_entry.active_trials.extend([t.id for t in trials])

            worker_entry.ref.initial_worker_queue.remote(
                trials[:num_cpus],
                trials[num_cpus:] if num_gpus > 0 else None,
            )

        self.trial_manager.display_trial_result.remote()

    def assign_trial(self, worker_id: int, worker_type: WorkerType) -> None:
        match worker_type:
            case WorkerType.CPU:
                cpu_scheduling(
                    self.trial_manager,
                    self.worker_manager,
                    worker_id,
                    self.logger,
                )
            case WorkerType.GPU:
                cpu_scheduling(
                    self.trial_manager,
                    self.worker_manager,
                    worker_id,
                    self.logger,
                )

    def run(self) -> None:
        """
        開始訓練過程, 將試驗分配給工作者並處理完成的結果。

        該方法會持續運行直到所有的試驗都完成。
        """
        self.logger.info("訓練開始")
        self.init_workers()
        self.finish_event.wait()

        self.trial_manager.print_iteration_count.remote()
        self.logger.info("🎉 所有 Trial 訓練完成!")
        self.worker_manager.stop_all_workers()

    def get_workers_logs(self) -> None:
        """
        獲取所有工作者的日誌並將其保存到文件中。
        該方法會將每個工作者的日誌寫入到相應的文件中。
        """
        log_dir = None
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_dir = Path(handler.baseFilename).parent  # 取得資料夾路徑
                break

        if log_dir is None:
            self.logger.error("logs檔案資料夾不存在")
            return

        for worker_entry in self.worker_manager.workers.values():
            worker = worker_entry.ref

            future = ray.get(worker.get_log_file.remote())  # type: ignore[reportGeneralTypeIssues]
            with (Path(log_dir) / f"worker-{future['id']}.log").open("w") as f:
                f.write(future["content"])
