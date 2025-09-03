import logging
import random
from pathlib import Path

import ray
import torch
from ray.actor import ActorHandle
from torch import nn, optim
from torch.utils.data import DataLoader

from .config import (
    ITERATION_PER_GENERATION,
)
from .trial_state import TrialState
from .utils import (
    Checkpoint,
    DataloaderFactory,
    TrainStepFunction,
    TrialStatus,
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

        record.trial_id = getattr(record, "trial_id", "N/A")
        return super().format(record)


def get_worker_logger(worker_id: int, worker_type: WorkerType) -> logging.Logger:
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

        match worker_type:
            case WorkerType.GPU:
                worker_type_str = "GPU"
            case WorkerType.CPU:
                worker_type_str = "CPU"

        formatter = WorkerLoggerFormatter(
            f"[%(asctime)s] %(levelname)s {worker_type_str} "
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


@ray.remote
class Worker:
    """
    表示一個 worker 節點, 負責訓練與回報試驗結果。

    Attributes:
        worker_state (WorkerState): Worker 的狀態資訊。
        active_trials (dict): 活躍試驗的字典。
        train_step (TrainStepFunction): 執行訓練步驟的函式。
        device (torch.device): 使用的設備 (CPU 或 GPU) 。
        logger (logging.Logger): 負責日誌紀錄。
    """

    def __init__(
        self,
        worker_state: WorkerState,
        train_step: TrainStepFunction,
        tuner: ActorHandle,
        trial_manager: ActorHandle,
        dataloader_factory: DataloaderFactory,
    ) -> None:
        """
        初始化 Worker, 設定狀態與參數。

        Args:
            worker_state (WorkerState): Worker 的狀態資訊。
            train_step (TrainStepFunction): 訓練步驟函式。
            tuner (ActorHandle): 負責接收訓練結果的 Actor。
        """
        self.worker_state: WorkerState = worker_state
        self.active_trials: dict[int, TrialState] = {}
        self.train_step: TrainStepFunction = train_step
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = get_worker_logger(
            worker_id=worker_state.id,
            worker_type=worker_state.worker_type,
        )
        self.tuner: ActorHandle = tuner
        self.trial_manager: ActorHandle = trial_manager
        self.iteration_per_generation: int = ITERATION_PER_GENERATION
        self.interrupt_set: set = set()
        self.dataloader_factory: DataloaderFactory = dataloader_factory
        self.is_stop: bool = False
        self.saved_checkpoint: dict[int, Checkpoint] = {}
        self.logger.info("初始化完成")

    def stealing_trial(self, trial_id: int) -> None:
        trial_state = self.active_trials.get(trial_id, None)

        if trial_state is None:
            self.log("info", "嘗試偷取的 Trial 不存在", trial_id=trial_id)
            return

        self.interrupt_set.add(trial_id)

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

    @timer()
    def assign_trial(self, trial_state: TrialState) -> None:
        """
        將試驗分配給該 worker 並開始訓練。

        Args:
            trial_state (TrialState): 試驗狀態。

        Returns:
            TrialState: 更新後的試驗狀態。
        """
        if len(self.active_trials) >= self.worker_state.max_trials:
            return

        self.log("info", "接收到新的 Trial", trial_id=trial_state.id)
        self._trial_load_checkpoint(trial_state)
        self.save_checkpoint(trial_state)
        trial_state.update_worker_state(self.worker_state)
        self.active_trials[trial_state.id] = trial_state

    def init_trial_queue(self, trials: list[TrialState]) -> None:
        for trial_state in trials:
            self.active_trials[trial_state.id] = trial_state

    def run(self) -> None:
        while not self.is_stop:
            trial_state = min(
                self.active_trials.values(),
                key=lambda x: x.generation,
                default=None,
            )

            if trial_state is None:
                continue

            ray.get(
                self.trial_manager.transition_status.remote(  # type: ignore[reportGeneralTypeIssues]
                    trial_state.id,
                    TrialStatus.RUNNING,
                    {
                        "worker_id": self.worker_state.id,
                        "worker_type": self.worker_state.worker_type,
                        "last_checkpoint_location": trial_state.last_checkpoint_location,  # noqa: E501
                    },
                ),
            )

            self.log(
                "info",
                f"開始訓練, target_generation: {trial_state.target_generation}",
                trial_id=trial_state.id,
            )

            hyper = trial_state.hyperparameter

            train_loader, test_loader, _ = self.dataloader_factory(
                hyper.batch_size,
                num_workers=0,
            )

            model, optimizer = trial_state.model_init_fn(self.device)

            self.train(trial_state, model, optimizer, train_loader, test_loader)
            self.active_trials.pop(trial_state.id)

    def train(
        self,
        trial_state: TrialState,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        test_loader: DataLoader,
    ) -> None:
        raw_gen = trial_state.target_generation
        target_generation = max(int(raw_gen), 1)

        if int(raw_gen) >= 1:
            iterations = self.iteration_per_generation
        else:
            iterations = max(
                1,
                int(self.iteration_per_generation * raw_gen),
            )

        self.log(
            "info",
            f"更新世代數: {target_generation}, 每世代迭代次數: {iterations}",
            trial_id=trial_state.id,
        )

        # ───────────────────────────── 開始執行訓練 ─────────────────────────────
        for _ in range(target_generation):
            for _ in range(iterations):
                if trial_state.generation >= trial_state.max_generation:
                    break

                if trial_state.id in self.interrupt_set:
                    return

                self.train_step(
                    model,
                    optimizer,
                    train_loader,
                    trial_state.hyperparameter.batch_size,
                    self.device,
                )

                trial_state.device_iteration_count[self.worker_state.worker_type] += 1

            if trial_state.id in self.interrupt_set:
                return

            trial_state.generation += 1
            trial_state.accuracy = self.test(model, test_loader)

            self.log(
                "info",
                f"Generation: {trial_state.generation} "
                f"Accuracy: {trial_state.accuracy}",
                trial_id=trial_state.id,
            )

            # ── 訓練結束 ──────────────────────────────────────────────────────────
            if (
                trial_state.generation >= trial_state.max_generation
                or trial_state.accuracy >= trial_state.stop_accuracy
            ):
                trial_state.update_checkpoint(model, optimizer)
                self.tuner.on_trial_complete.remote(
                    self.worker_state.id,
                    trial_state.id,
                    self.worker_state.worker_type,
                    {
                        "accuracy": trial_state.accuracy,
                        "generation": trial_state.generation,
                        "checkpoint": trial_state.checkpoint,
                        "device_iteration_count": trial_state.device_iteration_count,
                    },
                )
                return

            # ── 突變檢查 ──────────────────────────────────────────────────────────
            baseline = ray.get(self.trial_manager.get_mutation_baseline.remote())  # type: ignore[reportGeneralTypeIssues]
            mutation_ratio = 0.25

            if trial_state.accuracy <= baseline and random.random() >= mutation_ratio:
                self.log(
                    "info",
                    f"Baseline: {baseline}, Accuracy: {trial_state.accuracy}",
                    trial_id=trial_state.id,
                )
                trial_state.update_checkpoint(model, optimizer)
                self.tuner.on_trial_need_mutation.remote(
                    self.worker_state.id,
                    trial_state.id,
                    self.worker_state.worker_type,
                    {
                        "accuracy": trial_state.accuracy,
                        "generation": trial_state.generation,
                        "device_iteration_count": trial_state.device_iteration_count,
                    },
                )
                self.pop_checkpoint(trial_state.id)
                return

            # ── 更新 Trial ────────────────────────────────────────────────────────
            self.trial_manager.update_trial.remote(
                trial_state.id,
                {
                    "accuracy": trial_state.accuracy,
                    "generation": trial_state.generation,
                    "device_iteration_count": trial_state.device_iteration_count,
                },
            )

        # ── 目標世代數已達成, 更新檢查點並回報結果 ────────────────────────────
        trial_state.update_checkpoint(model, optimizer)
        self.tuner.on_trial_step_complete.remote(
            self.worker_state.id,
            trial_state.id,
            self.worker_state.worker_type,
            {
                "accuracy": trial_state.accuracy,
                "generation": trial_state.generation,
                "checkpoint": trial_state.checkpoint,
                "device_iteration_count": trial_state.device_iteration_count,
            },
        )

    def test(self, model: nn.Module, test_loader: DataLoader) -> float:
        """
        使用測試資料對模型進行測試並回傳準確率。

        Args:
            model (torch.nn.Module): 已訓練的模型。
            test_loader (torch.utils.data.DataLoader): 測試資料載入器。

        Returns:
            Accuracy: 模型測試結果的準確率。
        """
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for raw_inputs, raw_targets in test_loader:
                inputs, targets = (
                    raw_inputs.to(self.device),
                    raw_targets.to(self.device),
                )
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return correct / total

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

    def log(self, level: str, message: str, trial_id: int | str = "N/A") -> None:
        """
        根據指定的 log 級別輸出訊息。

        Args:
            level (str): 記錄等級 (info/debug/warning/error/critical) 。
            message (str): 要記錄的訊息。
            trial_id (Union[int, str], optional): 試驗 ID。預設為 "N/A"。
        """
        extra = {"trial_id": trial_id}
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
