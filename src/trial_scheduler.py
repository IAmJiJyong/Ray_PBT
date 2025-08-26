import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event

import ray
from ray import ObjectRef
from ray.actor import ActorHandle

from src.config import CPU_TRIALS_LIMIT, GPU_TRIALS_LIMIT

from .utils import TrialStatus, WorkerType
from .worker_manager import WorkerManager


def gpu_scheduling(
    worker_id: int,
    trial_manager: ActorHandle,
    worker_manager: WorkerManager,
    logger: logging.Logger,
) -> None:
    # 若沒有任何 pending 的 Trial,結束
    if not ray.get(trial_manager.has_pending_trials.remote()):  # type: ignore[reportGeneralTypeIssues]
        logger.info("沒有待分配的 Trial")
        return

    worker_entry = worker_manager.gpu_workers[worker_id]

    selected_trial = ray.get(
        trial_manager.acquire_pending_trial_for_gpu.remote(worker_id),  # type: ignore[reportGeneralTypeIssues]
    )

    worker_manager.assign_trial_to_worker(
        worker_entry.id,
        selected_trial,
    )


def cpu_scheduling(
    worker_id: int,
    trial_manager: ActorHandle,
    worker_manager: WorkerManager,
    logger: logging.Logger,
) -> None:
    if not ray.get(trial_manager.has_pending_trials.remote()):  # type: ignore[reportGeneralTypeIssues]
        logger.info("沒有待分配的 Trial")
        return

    worker_entry = worker_manager.cpu_workers[worker_id]

    target_trial = ray.get(
        trial_manager.acquire_pending_trial_for_cpu.remote(  # type: ignore[reportGeneralTypeIssues]
            worker_id,
            len(worker_manager.cpu_workers),
        ),
    )

    worker_manager.assign_trial_to_worker(
        worker_entry.id,
        target_trial,
    )


def stealing_strategy(
    worker_manager: WorkerManager,
    trial_manager: ActorHandle,
    logger: logging.Logger,
    interrupted_record_set: set[tuple[int, int]],
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

    trial_id = worker.active_trials[0]

    logger.info("嘗試從 CPU Worker %d 偷取 Trial %d", worker.id, trial_id)
    worker.ref.stealing_trial.remote(trial_id)  # type: ignore[reportGeneralTypeIssues])
    worker_manager.release_slots(worker.id, trial_id)
    ray.get(trial_manager.transition_status.remote(trial_id, TrialStatus.PENDING))  # type: ignore[reportGeneralTypeIssues]

    interrupted_record_set.add((worker.id, trial_id))


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
        stream_handler.setLevel(logging.INFO)  # 只顯示 INFO 級別以上的訊息
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

        self.logger.info("初始化完成")
        self._finish_event = Event()

        self.interrupted_record_set: set[tuple[int, int]] = set()

    def init_worker_queue(self) -> None:
        for worker_entry in self.worker_manager.gpu_workers.values():
            trials = ray.get(
                self.trial_manager.acquire_pending_trials.remote(
                    worker_entry.id,
                    GPU_TRIALS_LIMIT,
                    WorkerType.GPU,
                ),  # type: ignore[reportGeneralTypeIssues]
            )
            self.worker_manager.assign_trials_to_worker(worker_entry.id, trials)

        for worker_entry in self.worker_manager.cpu_workers.values():
            trials = ray.get(
                self.trial_manager.acquire_pending_trials.remote(
                    worker_entry.id,
                    CPU_TRIALS_LIMIT,
                    WorkerType.CPU,
                ),  # type: ignore[reportGeneralTypeIssues]
            )
            self.worker_manager.assign_trials_to_worker(worker_entry.id, trials)

    def assign_trial_to_worker(self, worker_id: int, worker_type: WorkerType) -> None:  # type: ignore[reportGeneralTypeIssues]
        """
        將一個試驗分配給一個可用的工作者。

        如果所有工作者都忙碌, 則返回當前正在運行的訓練任務。

        Returns:
            List[ObjectRef]: 當前正在運行的訓練任務列表。
        """
        has_pending_trials = ray.get(
            self.trial_manager.has_pending_trials.remote(),  # type: ignore[reportGeneralTypeIssues]
        )
        match worker_type:
            case WorkerType.CPU:
                if has_pending_trials:
                    cpu_scheduling(
                        worker_id,
                        self.trial_manager,
                        self.worker_manager,
                        self.logger,
                    )
            case WorkerType.GPU:
                if not has_pending_trials:
                    stealing_strategy(
                        self.worker_manager,
                        self.trial_manager,
                        self.logger,
                        self.interrupted_record_set,
                    )
                gpu_scheduling(
                    worker_id,
                    self.trial_manager,
                    self.worker_manager,
                    self.logger,
                )

    def run(self) -> None:
        """
        開始訓練過程, 將試驗分配給工作者並處理完成的結果。

        該方法會持續運行直到所有的試驗都完成。
        """
        self.logger.info("訓練開始")
        self.init_worker_queue()

        self._finish_event.wait()

        self.trial_manager.print_iteration_count.remote()
        self.logger.info("🎉 所有 Trial 訓練完成!")
        self.worker_manager.stop_all_workers()

    def finish(self) -> None:
        self._finish_event.set()

    def is_interrupted(self, worker_id: int, trial_id: int) -> bool:
        return (worker_id, trial_id) in self.interrupted_record_set

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
