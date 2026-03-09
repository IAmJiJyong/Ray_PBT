import copy
from dataclasses import dataclass, field
from typing import TypedDict

import ray
import torch
from torch import nn, optim

from .config import (
    MAX_GENERATION,
    STOP_ACCURACY,
)
from .utils import (
    Checkpoint,
    CheckpointLocation,
    Hyperparameter,
    TrialStatus,
    WorkerState,
    WorkerType,
)

ALLOWED_PARTIAL_KEYS = {
    "accuracy",
    "checkpoint",
    "chunk_size",
    "device_iteration_count",
    "generation",
    "hyperparameter",
    "last_checkpoint_location",
    "status",
    "worker_id",
    "worker_type",
}


class PartialTrialState(TypedDict, total=False):
    """
    用於表示 trial 狀態的部分資訊 (Partial State), 允許缺少部分欄位。
    這個結構主要用於在更新或檢查 trial 狀態時, 避免需要完整定義所有欄位。

    Attributes:
        accuracy (float): 該 trial 的當前準確率 (accuracy)。
        checkpoint (Checkpoint): 儲存當前模型或訓練進度的 Checkpoint。
        chunk_size (float): 在資料分批 (batch/chunk) 過程中的大小設定。
        generation (int): 該 trial 所屬的世代 (generation), 用於 PBT 演化流程。
        hyperparameter (Hyperparameter): 該 trial 使用的超參數 (hyperparameters)。
        last_checkpoint_location (CheckpointLocation): 最近一次儲存的 Checkpoint 位置資訊。
        status (TrialStatus): 該 trial 當前的執行狀態 (running, completed, failed 等)。
        worker_id (int): 負責執行此 trial 的 worker 識別碼 (ID)。
        worker_type (WorkerType): 該 worker 的類型 (CPU 或 GPU)。
    """

    accuracy: float
    checkpoint: Checkpoint
    chunk_size: float
    device_iteration_count: dict[WorkerType, int]
    generation: int
    hyperparameter: Hyperparameter
    last_checkpoint_location: CheckpointLocation
    status: TrialStatus
    worker_id: int
    worker_type: WorkerType | None


@dataclass(slots=True)
class TrialState:
    id: int
    hyperparameter: Hyperparameter

    max_generation: int = MAX_GENERATION
    status: TrialStatus = TrialStatus.PENDING
    worker_id: int = -1
    worker_type: WorkerType | None = None
    run_time: float = 0
    generation: int = 0
    device_iteration_count: dict[WorkerType, int] = field(
        default_factory=lambda: {WorkerType.CPU: 0, WorkerType.GPU: 0},
    )
    checkpoint: Checkpoint = field(default_factory=Checkpoint.empty)
    last_checkpoint_location: CheckpointLocation = field(
        default_factory=CheckpointLocation.empty,
    )
    accuracy: float = 0.0
    stop_accuracy: float = STOP_ACCURACY
    target_generation: float = 1

    def update_worker_state(self, worker_state: WorkerState) -> None:
        self.worker_type = worker_state.worker_type
        self.worker_id = worker_state.id
        self.last_checkpoint_location = CheckpointLocation(
            worker_state.id,
            ray.get_runtime_context().current_actor,
        )

    def update_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        self.checkpoint.model_state_dict = model.cpu().state_dict()
        optimizer_state_dict = optimizer.state_dict()

        for state in optimizer_state_dict["state"].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        self.checkpoint.optimizer_state_dict = optimizer_state_dict

    def update_partial(self, partial: PartialTrialState) -> None:
        for key, value in partial.items():
            if key in ALLOWED_PARTIAL_KEYS:
                setattr(self, key, value)
            else:
                msg = f"無法更新 TrialState 屬性 '{key}'"
                raise AttributeError(msg)

    def get_remote_checkpoint(self) -> Checkpoint:
        if self.last_checkpoint_location.is_empty():
            return Checkpoint.empty()

        return ray.get(
            self.last_checkpoint_location.worker_reference.get_checkpoint.remote(  # type:ignore[reportGeneralTypeIssues]
                self.id,
            ),
        )

    def pop_remote_checkpoint(self) -> None:
        if self.last_checkpoint_location.is_empty():
            return
        ray.get(
            self.last_checkpoint_location.worker_reference.pop_checkpoint.remote(  # type:ignore[reportGeneralTypeIssues]
                self.id,
            ),
        )

    def remove_remote_checkpoint(self) -> None:
        if self.last_checkpoint_location.is_empty():
            return
        self.last_checkpoint_location.worker_reference.remove_checkpoint.remote(  # type:ignore[reportGeneralTypeIssues]
            self.id,
        )

    @property
    def snapshot(self) -> "TrialState":
        new_trial = copy.copy(self)
        new_trial.checkpoint = Checkpoint.empty()
        return new_trial

    def set_target_generation(self, target_generation: int) -> None:
        if target_generation < 1:
            msg = "Chunk size must be at least 1"
            raise ValueError(msg)
        self.target_generation = target_generation
