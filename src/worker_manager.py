import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import count
from pathlib import Path

import ray
from ray.actor import ActorHandle

from .trial_state import TrialState
from .utils import (
    DataloaderFactory,
    TrainStepFunction,
    WorkerState,
    WorkerType,
    get_head_node_address,
)
from .worker import Worker


def get_worker_manager_logger() -> logging.Logger:
    timestamp = (datetime.now(UTC) + timedelta(hours=8)).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path.cwd() / "logs" / timestamp
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("WorkerManager")

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # 或者選擇更合適的級別

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s WORKER_MANAGER -- %(message)s",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)  # 只顯示 INFO 級別以上的訊息
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(Path(log_dir) / "worker_manager.log")
        file_handler.setLevel(logging.DEBUG)  # 記錄所有級別的日誌
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


@dataclass
class WorkerEntry:
    state: WorkerState
    ref: ActorHandle
    active_trials: list[int] = field(default_factory=list)

    @property
    def id(self) -> int:
        return self.state.id


class WorkerManager:
    def __init__(
        self,
        tuner: ActorHandle,
        trial_manager: ActorHandle,
        train_step: TrainStepFunction,
        dataloader_factory: DataloaderFactory,
    ) -> None:
        self.workers: dict[int, WorkerEntry] = {}
        self.assign_count: dict[str, int] = {"assign": 0, "locality": 0}
        self.logger: logging.Logger = get_worker_manager_logger()

        worker_states = generate_all_worker_states()
        for worker_state in worker_states:
            worker_ref: ActorHandle = Worker.options(  # type: ignore[reportGeneralTypeIssues]
                max_concurrency=5,
                name=f"worker-{worker_state.id}",
                num_cpus=1,
                resources={worker_state.node_name: 0.01},
            ).remote(
                worker_state,
                train_step,
                tuner,
                trial_manager,
                dataloader_factory,
            )

            self.workers[worker_state.id] = WorkerEntry(worker_state, worker_ref)

    def assign_trial_to_worker(
        self,
        worker_type: WorkerType,
        worker_id: int,
        trial: TrialState,
    ) -> None:
        entry = self.workers.get(worker_id, None)
        if entry is None:
            msg = f"Worker {worker_id} 不存在."
            raise ValueError(msg)

        entry.active_trials.append(trial.id)
        self.assign_count["assign"] += 1

        if trial.last_checkpoint_location.worker_id == worker_id:
            self.logger.info(
                "分配 Trial %d snapshot 到 Worker %d",
                trial.id,
                worker_id,
            )
            self.assign_count["locality"] += 1

            match worker_type:
                case WorkerType.CPU:
                    entry.ref.add_cpu_task.remote(trial.snapshot)  # type: ignore[reportGeneralTypeIssues]
                case WorkerType.GPU:
                    entry.ref.add_gpu_task.remote(trial.snapshot)  # type: ignore[reportGeneralTypeIssues]

        else:
            self.logger.info(
                "分配 Trial %d 到 Worker %d",
                trial.id,
                worker_id,
            )
            match worker_type:
                case WorkerType.CPU:
                    entry.ref.add_cpu_task.remote(trial)  # type: ignore[reportGeneralTypeIssues]
                case WorkerType.GPU:
                    entry.ref.add_gpu_task.remote(trial)  # type: ignore[reportGeneralTypeIssues]

    def release_slots(self, worker_id: int, trial_id: int) -> None:
        entry = self.workers.get(worker_id, None)
        if entry is None:
            msg = f"Worker {worker_id} 不存在."
            raise ValueError(msg)

        try:
            entry.active_trials.remove(trial_id)
        except ValueError:
            msg = f"Trial {trial_id} 不存在於 {entry.active_trials}"
            self.logger.exception(msg)

    def stop_all_workers(self) -> None:
        ray.get(
            [worker_entry.ref.stop.remote() for worker_entry in self.workers.values()],  # type:ignore[reportGeneralTypeIssues]
        )

    def get_log_file(self) -> str:
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
            self.logger.error("Logs direction is not exists")
            return ""

        with Path(log_dir).open("r") as f:
            return f.read()


def generate_all_worker_states() -> list[WorkerState]:
    """
    根據 Ray 叢集的節點資源建立所有 Worker。


    Returns:
        List[ActorHandle]: 建立的 Worker Actor 清單。
    """

    visited_address = set()
    worker_states: list[WorkerState] = []
    count_gen = count(start=0, step=1)
    head_node_address = get_head_node_address()

    for node in ray.nodes():
        node_address = node["NodeManagerAddress"]
        if node["Alive"]:
            if node_address in visited_address:
                continue

            resource = node["Resources"]

            gpus = int(resource.get("GPU", 0))
            cpus = int(resource.get("CPU"))
            cpus = cpus - gpus - 1

            if node_address == head_node_address:
                cpus -= 2

            worker_states.append(
                WorkerState(
                    id=next(count_gen),
                    num_cpus=cpus,
                    num_gpus=gpus,
                    node_name=f"node:{node_address}",
                ),
            )
            visited_address.add(node_address)

    return worker_states
