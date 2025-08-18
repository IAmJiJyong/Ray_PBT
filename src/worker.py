import heapq
import logging
import random
from collections.abc import Callable
from itertools import count
from pathlib import Path

import ray
import torch
from ray.actor import ActorHandle
from torch import nn, optim
from torch.types import Device
from torch.utils.data import DataLoader

from .config import (
    ITERATION_PER_GENERATION,
)
from .trial_state import TrialState
from .utils import (
    Checkpoint,
    DataloaderFactory,
    TrainStepFunction,
    WorkerState,
    WorkerType,
    timer,
)


class WorkerLoggerFormatter(logging.Formatter):
    """
    自訂的日誌格式器, 用於在日誌中加入 worker_id 和 trial_id。

    Attributes:
        None
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日誌紀錄, 加入 worker_id 和 trial_id。

        Args:
            record (logging.LogRecord): 日誌紀錄。

        Returns:
            str: 格式化後的日誌訊息。
        """

        worker_type = getattr(record, "worker_type", WorkerType.CPU)
        if worker_type == WorkerType.GPU:
            record.worker_type = "GPU"
        elif worker_type == WorkerType.CPU:
            record.worker_type = "CPU"

        record.trial_id = getattr(record, "trial_id", "N/A")
        return super().format(record)


def get_worker_logger(worker_id: int) -> logging.Logger:
    """
    建立並回傳指定 worker_id 的 logger, 支援終端輸出與檔案寫入。

    Args:
        worker_id (int): Worker 的唯一識別碼。

    Returns:
        logging.Logger: 設定完成的 logger。
    """

    log_dir = Path(Path.cwd()) / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"worker-{worker_id}")

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        formatter = WorkerLoggerFormatter(
            f"[%(asctime)s] %(levelname)s %(worker_type)s "
            f"WORKER_ID: {worker_id} TRIAL_ID: %(trial_id)s -- %(message)s",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(Path(log_dir) / f"worker-{worker_id}.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class WorkerResource:
    def __init__(self, max_cpus: int, max_gpus: int) -> None:
        self.max_cpus: int = max_cpus
        self.max_gpus: int = max_gpus
        self.available_cpus: int = self.max_cpus
        self.available_gpus: int = self.max_gpus

    def decide_cpu_allocation(self, trial_state: TrialState) -> int:
        progress = trial_state.generation / trial_state.max_generation
        early_phase = 0.3
        middle_phase = 0.7
        if progress <= early_phase:
            return min(4, self.max_cpus)

        if progress <= middle_phase:
            return min(8, self.max_cpus)
        return self.max_cpus

    def request_cpu(self, num_cpus: int) -> bool:
        if self.available_cpus <= 0 or self.available_cpus < num_cpus:
            return False
        self.available_cpus -= num_cpus
        return True

    def release_cpu(self, num_cpus: int) -> None:
        self.available_cpus += num_cpus

    def request_gpu(self, num_gpus: int = 1) -> bool:
        if self.available_gpus <= 0 or self.available_gpus < num_gpus:
            return False
        self.available_gpus -= num_gpus
        return True

    def release_gpu(self, num_gpus: int = 1) -> None:
        self.available_gpus += num_gpus


@ray.remote
def run_task(  # noqa: PLR0913
    trial_state: TrialState,
    worker_type: WorkerType,
    num_cores: int,
    train_step: Callable,
    dataloader_factory: DataloaderFactory,
    trial_manager: ActorHandle,
    worker: ActorHandle,
) -> None:
    torch.set_num_threads(num_cores)

    hyperparameter = trial_state.hyperparameter
    train_loader, test_loader, _ = dataloader_factory(
        hyperparameter.batch_size,
        num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, optimizer = trial_state.model_init_fn(device)

    trial_manager.transition_to_running.remote(
        trial_state.id,
        {"worker_type": worker_type, "num_cores": num_cores},
    )

    train(
        trial_state,
        num_cores,
        train_step,
        device,
        worker_type,
        model,
        optimizer,
        train_loader,
        test_loader,
        trial_manager,
        worker,
    )


def train(  # noqa: PLR0913
    trial_state: TrialState,
    num_cores: int,
    train_step: Callable,
    device: Device,
    worker_type: WorkerType,
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    trial_manager: ActorHandle,
    worker: ActorHandle,
) -> None:
    for _ in range(trial_state.chunk_size):
        for _ in range(ITERATION_PER_GENERATION):
            if trial_state.generation >= trial_state.max_generation:
                break

            train_step(
                model,
                optimizer,
                train_loader,
                trial_state.hyperparameter.batch_size,
                device,
            )

            trial_state.device_iteration_count[worker_type] += 1

        trial_state.generation += 1
        trial_state.accuracy = test(device, model, test_loader)

        trial_manager.update_trial.remote(
            trial_state.id,
            {
                "accuracy": trial_state.accuracy,
                "generation": trial_state.generation,
            },
        )

        if (
            trial_state.generation >= trial_state.max_generation
            or trial_state.accuracy >= trial_state.stop_accuracy
        ):
            trial_state.update_checkpoint(model, optimizer)
            worker.on_task_complete.remote(
                trial_state,
                worker_type,
                num_cores,
            )
            return

        baseline = ray.get(trial_manager.get_mutation_baseline.remote())  # type: ignore[reportGeneralTypeIssues]
        mutation_ratio = 0.25

        if trial_state.accuracy <= baseline and random.random() >= mutation_ratio:
            trial_state.update_checkpoint(model, optimizer)
            worker.on_task_need_mutation.remote(trial_state, worker_type, num_cores)
            return

    trial_state.update_checkpoint(model, optimizer)
    worker.on_task_step_complete.remote(trial_state, worker_type, num_cores)


def test(device: Device, model: nn.Module, test_loader: DataLoader) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for raw_inputs, raw_targets in test_loader:
            inputs, targets = (
                raw_inputs.to(device),
                raw_targets.to(device),
            )
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return correct / total


@ray.remote
class Worker:
    def __init__(
        self,
        worker_state: WorkerState,
        train_step: TrainStepFunction,
        tuner: ActorHandle,
        trial_manager: ActorHandle,
        dataloader_factory: DataloaderFactory,
    ) -> None:
        self.worker_state: WorkerState = worker_state
        self.train_step: TrainStepFunction = train_step
        self.tuner: ActorHandle = tuner
        self.trial_manager: ActorHandle = trial_manager
        self.dataloader_factory: DataloaderFactory = dataloader_factory

        self.interrupt_set: set = set()
        self.iteration_per_generation: int = ITERATION_PER_GENERATION
        self.logger: logging.Logger = get_worker_logger(
            worker_id=worker_state.id,
        )
        self.saved_checkpoint: dict[int, Checkpoint] = {}
        self.logger.info("初始化完成")
        self.is_stop: bool = False
        self.cpu_task_queue: list[tuple[int, int, TrialState]] = []
        self.gpu_task_queue: list[tuple[int, int, TrialState]] = []
        self._task_counter: count = count()
        self.worker_resource: WorkerResource = WorkerResource(
            self.worker_state.num_cpus,
            self.worker_state.num_gpus,
        )

    def save_checkpoint(self, trial_state: TrialState) -> None:
        self.log("info", "儲存 Checkpoint", trial_id=trial_state.id)
        if trial_state.checkpoint.is_empty():
            self.log("warning", "Checkpoint 為空", trial_id=trial_state.id)
            return

        self.saved_checkpoint[trial_state.id] = trial_state.checkpoint

    def get_checkpoint(self, trial_id: int) -> Checkpoint:
        """
        取得指定試驗的檢查點。

        Args:
            trial_id (int): 試驗 ID。
        """
        self.log("info", "取得 Checkpoint", trial_id=trial_id)
        return self.saved_checkpoint.get(trial_id, Checkpoint.empty())

    def pop_checkpoint(self, trial_id: int) -> Checkpoint:
        """
        取得並移除指定試驗的檢查點。

        Args:
            trial_id (int): 試驗 ID。
        """
        self.log("info", "取得並移除 Checkpoint", trial_id=trial_id)
        return self.saved_checkpoint.pop(trial_id, Checkpoint.empty())

    def remove_checkpoint(self, trial_id: int) -> None:
        """
        移除指定試驗的檢查點。

        Args:
            trial_id (int): 試驗 ID。
        """
        if trial_id in self.saved_checkpoint:
            self.saved_checkpoint.pop(trial_id)
            self.log("info", f"已移除 Trial {trial_id} 的檢查點", trial_id=trial_id)
        else:
            self.log("warning", f"Trial {trial_id} 的檢查點不存在", trial_id=trial_id)

    def on_task_complete(
        self,
        trial_state: TrialState,
        worker_type: WorkerType,
        num_cores: int,
    ) -> None:
        self.log(
            "info",
            "Trial 訓練完成",
            worker_type=worker_type,
            trial_id=trial_state.id,
        )
        self.tuner.on_trial_complete.remote(
            self.worker_state.id,
            trial_state.id,
            {
                "accuracy": trial_state.accuracy,
                "checkpoint": trial_state.checkpoint,
                "generation": trial_state.generation,
            },
        )

        match worker_type:
            case WorkerType.CPU:
                self.worker_resource.release_cpu(num_cores)
                self.dispatch_cpu_task()
            case WorkerType.GPU:
                self.worker_resource.release_gpu(num_cores)
                self.dispatch_gpu_task()

    def on_task_need_mutation(
        self,
        trial_state: TrialState,
        worker_type: WorkerType,
        num_cores: int,
    ) -> None:
        self.tuner.on_trial_need_mutation.remote(
            self.worker_state.id,
            trial_state.id,
            {
                "accuracy": trial_state.accuracy,
                "checkpoint": trial_state.checkpoint,
                "generation": trial_state.generation,
            },
        )

        match worker_type:
            case WorkerType.CPU:
                self.worker_resource.release_cpu(num_cores)
                self.dispatch_cpu_task()
            case WorkerType.GPU:
                self.worker_resource.release_gpu(num_cores)
                self.dispatch_gpu_task()

    def on_task_step_complete(
        self,
        trial_state: TrialState,
        worker_type: WorkerType,
        num_cores: int,
    ) -> None:
        self.log(
            "info",
            f"Task 階段完成, num_cores: {num_cores}",
            worker_type=worker_type,
            trial_id=trial_state.id,
        )
        self.tuner.on_trial_step_complete.remote(
            self.worker_state.id,
            trial_state.id,
            {
                "accuracy": trial_state.accuracy,
                "checkpoint": trial_state.checkpoint,
                "generation": trial_state.generation,
            },
        )

        match worker_type:
            case WorkerType.CPU:
                self.worker_resource.release_cpu(num_cores)
                self.dispatch_cpu_task()
            case WorkerType.GPU:
                self.worker_resource.release_gpu(num_cores)
                self.dispatch_gpu_task()

    @timer()
    def _trial_load_checkpoint(self, trial_state: TrialState) -> None:
        """
        嘗試從檢查點載入試驗狀態。

        Args:
            trial_state (TrialState): 試驗狀態。
        """
        if trial_state.last_checkpoint_location.is_empty():
            return

        if trial_state.last_checkpoint_location.worker_id in self.saved_checkpoint:
            self.log("info", "載入本地 checkpoint", trial_id=trial_state.id)
            checkpoint = self.get_checkpoint(trial_state.id)
            if not checkpoint.is_empty():
                trial_state.checkpoint = checkpoint
                self.log(
                    "info",
                    "載入成功",
                    trial_id=trial_state.id,
                )
            else:
                self.log(
                    "warning",
                    "載入失敗, Checkpoint 為空",
                    trial_id=trial_state.id,
                )
        else:
            trial_state.remove_remote_checkpoint()

    def add_cpu_task(self, trial_state: TrialState) -> None:
        heapq.heappush(
            self.cpu_task_queue,
            (trial_state.generation, next(self._task_counter), trial_state),
        )
        self.dispatch_cpu_task()

    def add_gpu_task(self, trial_state: TrialState) -> None:
        heapq.heappush(
            self.gpu_task_queue,
            (trial_state.generation, next(self._task_counter), trial_state),
        )
        self.dispatch_cpu_task()

    def initial_worker_queue(
        self,
        cpu_trials: list[TrialState],
        gpu_trials: list[TrialState] | None = None,
    ) -> None:
        for trial in cpu_trials:
            heapq.heappush(
                self.cpu_task_queue,
                (trial.generation, next(self._task_counter), trial),
            )

        if gpu_trials:
            for trial in gpu_trials:
                heapq.heappush(
                    self.gpu_task_queue,
                    (trial.generation, next(self._task_counter), trial),
                )

        self.dispatch_gpu_task()
        self.dispatch_cpu_task()

    def dispatch_cpu_task(self) -> None:
        self.log(
            "info",
            f"Available CPU:{self.worker_resource.available_cpus}",
        )
        while self.worker_resource.available_cpus:
            if not self.cpu_task_queue:
                self.log("info", "CPU task queue is empty.")
                return

            peek_trial = self.cpu_task_queue[0][2]
            num_cpus = self.worker_resource.decide_cpu_allocation(peek_trial)

            if not self.worker_resource.request_cpu(num_cpus):
                return

            _, _, trial = heapq.heappop(self.cpu_task_queue)
            self.log(
                "info",
                f"訓練指派 CPU:{num_cpus}",
                trial_id=trial.id,
                worker_type=WorkerType.CPU,
            )
            run_task.options(  # type: ignore[reportGeneralTypeIssues]
                num_cpus=num_cpus,
                resources={self.worker_state.node_name: 0.01},
            ).remote(
                trial,
                WorkerType.CPU,
                num_cpus,
                self.train_step,
                self.dataloader_factory,
                self.trial_manager,
                ray.get_runtime_context().current_actor,
            )

    def dispatch_gpu_task(self) -> None:
        self.log(
            "info",
            f"Available GPU:{self.worker_resource.available_gpus}",
        )
        while self.worker_resource.available_gpus:
            if not self.gpu_task_queue:
                self.log("info", "GPU task queue is empty.")
                return

            num_gpus = 1
            if not self.worker_resource.request_gpu(num_gpus):
                return

            _, _, trial = heapq.heappop(self.gpu_task_queue)
            self.log(
                "info",
                f"訓練指派 GPU:{num_gpus}",
                trial_id=trial.id,
                worker_type=WorkerType.GPU,
            )
            run_task.options(  # type: ignore[reportGeneralTypeIssues]
                num_cpus=1,
                num_gpus=num_gpus,
                resources={self.worker_state.node_name: 0.01},
            ).remote(
                trial,
                WorkerType.GPU,
                num_gpus,
                self.train_step,
                self.dataloader_factory,
                self.trial_manager,
                ray.get_runtime_context().current_actor,
            )

    def get_log_file(self) -> dict[str, int | str]:
        """
        取得 worker 對應的日誌檔案內容。

        Returns:
            dict: 包含 worker ID 與對應日誌內容的字典。
        """
        log_dir = None
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_dir = handler.baseFilename
                break

        if not log_dir:
            self.log("error", "Logs direction is not exists")
            return {"id": self.worker_state.id, "content": ""}

        with Path(log_dir).open("r") as f:
            return {"id": self.worker_state.id, "content": f.read()}

    def log(
        self,
        level: str,
        message: str,
        trial_id: int | str = "N/A",
        worker_type: WorkerType = WorkerType.CPU,
    ) -> None:
        """
        根據指定的 log 級別輸出訊息。

        Args:
            level (str): 記錄等級 (info/debug/warning/error/critical) 。
            message (str): 要記錄的訊息。
            trial_id (Union[int, str], optional): 試驗 ID。預設為 "N/A"。
        """
        extra = {"trial_id": trial_id, "worker_type": worker_type}
        if level == "info":
            self.logger.info(message, extra=extra)
            return
        if level == "debug":
            self.logger.debug(message, extra=extra)
            return
        if level == "warning":
            self.logger.warning(message, extra=extra)
            return
        if level == "critical":
            self.logger.critical(message, extra=extra)
            return
        if level == "error":
            self.logger.error(message, extra=extra)
            return

    def stop(self) -> None:
        self.is_stop = True
