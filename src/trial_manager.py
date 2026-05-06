import heapq
import logging
import math
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import ray

from .config import (
    TRIAL_PROGRESS_OUTPUT_PATH,
)
from .trial_state import PartialTrialState, TrialState
from .utils import TrialStatus, WorkerState, WorkerType, colored_progress_bar

ALLOWED_TRANSITION: dict[TrialStatus, set[TrialStatus]] = {
    TrialStatus.PENDING: {TrialStatus.WAITING},
    TrialStatus.WAITING: {TrialStatus.RUNNING, TrialStatus.PENDING},
    TrialStatus.RUNNING: {
        TrialStatus.WAITING,
        TrialStatus.PENDING,
        TrialStatus.TERMINATED,
    },
    TrialStatus.TERMINATED: set(),
}


def get_trial_manager_logger() -> logging.Logger:
    timestamp = (datetime.now(UTC) + timedelta(hours=8)).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path.cwd() / "logs" / timestamp
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("TrialManager")

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # 或者選擇更合適的級別

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s TRIAL_MANAGER -- %(message)s",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)  # 只顯示 INFO 級別以上的訊息
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(Path(log_dir) / "trial_manager.log")
        file_handler.setLevel(logging.DEBUG)  # 記錄所有級別的日誌
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


@ray.remote
class TrialManager:
    def __init__(
        self,
        trial_states: list[TrialState],
    ) -> None:
        self.all_trials = {trial.id: trial for trial in trial_states}
        self.pending_ids = {trial.id for trial in trial_states}
        self.running_ids = set()
        self.completed_ids = set()
        self.waiting_ids = set()
        self.history_best: TrialState | None = None
        self.worker_states: list[WorkerState] = []

        self._mutation_baseline: float = 0.0
        self._upper_quantile_trials: list[TrialState] = []
        self.logger = get_trial_manager_logger()

    def set_worker_states(self, worker_states: list[WorkerState]) -> None:
        self.worker_states = worker_states

    def _get_trial_or_raise(self, trial_id: int) -> TrialState:
        trial = self.all_trials.get(trial_id)
        if trial is None:
            msg = f"Trial {trial_id} not found"
            raise ValueError(msg)
        return trial

    def _set_status(self, trial_id: int, new_status: TrialStatus) -> None:
        trial_state = self._get_trial_or_raise(trial_id)
        old_status = self.all_trials[trial_id].status

        allowed = ALLOWED_TRANSITION[old_status]
        if new_status not in allowed:
            msg = (
                f"Trial {trial_id} 錯誤的狀態轉移: "
                f"{old_status} -> {new_status}(僅允許: {allowed})"
            )
            raise ValueError(msg)

        trial_state.status = new_status
        self.logger.info("Trial %d 狀態從 %s -> %s", trial_id, old_status, new_status)

    def _transition_to_waiting(
        self,
        trial_id: int,
        partial: PartialTrialState | None = None,
    ) -> None:
        self._get_trial_or_raise(trial_id)

        if partial:
            self.update_trial(trial_id, partial)

        self._set_status(trial_id, TrialStatus.WAITING)
        self.pending_ids.discard(trial_id)
        self.waiting_ids.add(trial_id)

    def _transition_to_running(
        self,
        trial_id: int,
        partial: PartialTrialState | None = None,
    ) -> None:
        self._get_trial_or_raise(trial_id)

        if partial:
            self.update_trial(trial_id, partial)

        self._set_status(trial_id, TrialStatus.RUNNING)
        self.waiting_ids.discard(trial_id)
        self.running_ids.add(trial_id)

    def _transition_to_pending(
        self,
        trial_id: int,
        partial: PartialTrialState | None = None,
    ) -> None:
        self._get_trial_or_raise(trial_id)

        if partial:
            self.update_trial(trial_id, partial)

        self._set_status(trial_id, TrialStatus.PENDING)
        self.running_ids.discard(trial_id)
        self.pending_ids.add(trial_id)

    def _transition_to_completed(
        self,
        trial_id: int,
        partial: PartialTrialState | None = None,
    ) -> None:
        self._get_trial_or_raise(trial_id)

        if partial:
            self.update_trial(trial_id, partial)

        self._set_status(trial_id, TrialStatus.TERMINATED)
        self.running_ids.discard(trial_id)
        self.completed_ids.add(trial_id)

    def transition_status(
        self,
        trial_id: int,
        status: TrialStatus,
        partial: PartialTrialState | None = None,
    ) -> None:
        match status:
            case TrialStatus.PENDING:
                self._transition_to_pending(trial_id, partial)
            case TrialStatus.WAITING:
                self._transition_to_waiting(trial_id, partial)
            case TrialStatus.RUNNING:
                self._transition_to_running(trial_id, partial)
            case TrialStatus.TERMINATED:
                self._transition_to_completed(trial_id, partial)
            case _:
                msg = f"Unknown status: {status}"
                raise ValueError(msg)

    def acquire_pending_trials(
        self,
        worker_id: int,
        n: int,
        worker_type: WorkerType = WorkerType.CPU,
    ) -> list[TrialState]:
        acquired = []

        for trial_id in list(self.pending_ids)[:n]:
            trial = self.all_trials[trial_id]
            acquired.append(trial)
            self._transition_to_waiting(
                trial_id,
                {"worker_id": worker_id, "worker_type": worker_type},
            )

        return acquired

    def acquire_pending_trial_for_gpu(
        self,
        worker_id: int,
    ) -> TrialState | None:
        if not self.pending_ids:
            return None

        trials = self.get_pending_trials_with_min_iteration()
        if not trials:
            return None

        # Prioritize trials previously assigned to this worker and with checkpoint
        potential_trials = [
            t
            for t in trials
            if not t.last_checkpoint_location.is_empty()
            and t.last_checkpoint_location.worker_id == worker_id
        ]

        candidate_trial = potential_trials[0] if potential_trials else trials[0]

        # Retrieve the most up-to-date TrialState from all_trials
        current_trial_state = self.all_trials[candidate_trial.id]

        # Only proceed if the trial is truly PENDING
        if current_trial_state.status != TrialStatus.PENDING:
            self.logger.info(
                "Trial %d is no longer PENDING "
                "(current status: {current_trial_state.status}). "
                "Skipping acquisition.",
                current_trial_state.id,
            )
            return None

        # If it's PENDING, then we can proceed with the transition
        current_trial_state.set_target_generation(
            self.compute_target_generation(current_trial_state.generation),
        )

        self._transition_to_waiting(
            current_trial_state.id,
            {"worker_id": worker_id, "worker_type": WorkerType.GPU},
        )

        return current_trial_state

    def acquire_pending_trial_for_cpu(
        self,
        worker_id: int,
        k: int,
    ) -> TrialState | None:
        if not self.pending_ids:
            return None

        selected_trial = self.get_nlargest_iteration_trials(k)[-1]  # type: ignore[reportGeneralTypeIssues]
        selected_trial.set_target_generation(2)
        self._transition_to_waiting(
            selected_trial.id,
            {"worker_id": worker_id, "worker_type": WorkerType.CPU},
        )

        return selected_trial

    def get_pending_trials(self) -> list[TrialState]:
        return [self.all_trials[tid] for tid in self.pending_ids]

    def get_pending_trials_with_min_iteration(self) -> list[TrialState]:
        if not self.pending_ids:
            return []

        pending_trials = self.get_pending_trials()
        min_iter = min(pending_trials, key=lambda t: t.generation).generation
        return [trial for trial in pending_trials if trial.generation == min_iter]

    def get_least_iterated_pending_trial(self) -> TrialState | None:
        if not self.pending_ids:
            return None

        return min(
            (self.all_trials[tid] for tid in self.pending_ids),
            key=lambda t: t.generation,
            default=None,
        )

    def get_most_iterated_pending_trial(self) -> TrialState | None:
        if not self.pending_ids:
            return None

        return max(
            (self.all_trials[tid] for tid in self.pending_ids),
            key=lambda t: t.generation,
            default=None,
        )

    def compute_target_generation(self, generation: int) -> int:
        generations = sorted(
            [trial.generation for trial in self.all_trials.values()],
            reverse=True,
        )
        length = (len(generations) // 4) + 1
        target_generation = sum(generations[:length]) // length - generation + 1
        return max(target_generation, 1)

    def get_history_best_result(self) -> TrialState | None:
        return self.history_best

    def get_nlargest_iteration_trials(self, k: int) -> list[TrialState]:
        return heapq.nlargest(
            k,
            [
                trial
                for trial in self.all_trials.values()
                if trial.id in self.pending_ids
            ],
            key=lambda t: t.generation,
        )

    def get_mutation_baseline(
        self,
        ratio: float = 0.25,
    ) -> float:
        accuracy = [
            trial.accuracy for trial in self.all_trials.values() if trial.accuracy > 0
        ]
        quantile_size = math.ceil(len(self.all_trials) * ratio)

        result = heapq.nsmallest(
            quantile_size,
            accuracy,
        )

        if len(result) < quantile_size:
            return 0.0

        return result[-1]

    def get_cached_mutation_baseline(self) -> float:
        return self._mutation_baseline

    def get_cached_upper_quantile_trials(self) -> list[TrialState]:
        return self._upper_quantile_trials

    def get_uncompleted_trial_num(self) -> int:
        return len(self.all_trials) - len(self.completed_ids)

    def get_upper_quantile_trials(self, ratio: float = 0.25) -> list[TrialState]:
        trials = [trial for trial in self.all_trials.values() if trial.accuracy > 0]
        quantile_size = math.ceil(len(self.all_trials) * ratio)
        return heapq.nlargest(
            quantile_size,
            trials,
            key=lambda t: t.accuracy,
        )

    def has_pending_trials(self) -> bool:
        return bool(self.pending_ids)

    def maybe_update_mutation_baseline(self) -> None:
        self._mutation_baseline = self.get_mutation_baseline()
        self._upper_quantile_trials = self.get_upper_quantile_trials()

    def update_trial(self, trial_id: int, partial: PartialTrialState) -> None:
        trial_state = self._get_trial_or_raise(trial_id)

        old_checkpoint_location = trial_state.last_checkpoint_location

        trial_state.update_partial(partial)

        if (
            not old_checkpoint_location.is_empty()
            and old_checkpoint_location.worker_id
            and trial_state.last_checkpoint_location.worker_id
            != old_checkpoint_location.worker_id
        ):
            self.logger.info(
                "Trial %d 的檢查點位置已從 Worker %d 轉移到 Worker %d。"
                "通知舊 Worker 刪除其檢查點。",
                trial_id,
                old_checkpoint_location.worker_id,
                trial_state.last_checkpoint_location.worker_id,
            )

            if old_checkpoint_location.worker_reference:
                old_checkpoint_location.worker_reference.remove_checkpoint.remote(
                    trial_id,
                )

        if "accuracy" in partial:
            if trial_state.accuracy > 0 and (
                self.history_best is None
                or trial_state.accuracy > self.history_best.accuracy
            ):
                self.history_best = trial_state

            if self.history_best:
                self.logger.info(
                    "History best accuracy: %f, %s, iteration: %d",
                    self.history_best.accuracy,
                    str(self.history_best.hyperparameter),
                    self.history_best.generation,
                )
            self.maybe_update_mutation_baseline()
        self.display_trial_result()

    def is_finish(self) -> bool:
        self.logger.info(
            "已完成 Trial 數(%2d/%2d)",
            len(self.completed_ids),
            len(self.all_trials),
        )
        return len(self.completed_ids) >= len(self.all_trials)

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

    def mutation(self) -> PartialTrialState:
        upper_quantile = self.get_cached_upper_quantile_trials()
        chose_trial = random.choice(upper_quantile)
        hyperparameter = chose_trial.hyperparameter.explore()

        return {
            "hyperparameter": hyperparameter,
            "checkpoint": chose_trial.checkpoint,
        }

    def _worker_type_to_str(self, worker_type: WorkerType | None) -> str:
        match worker_type:
            case WorkerType.CPU:
                return "CPU"
            case WorkerType.GPU:
                return "GPU"
            case _:
                return ""

    def _worker_id_to_str(self, worker_id: int) -> str:
        if worker_id == -1:
            return ""
        return str(worker_id)

    def _save_at_to_str(self, trial: TrialState) -> str:
        if trial.last_checkpoint_location.is_empty():
            return ""
        return str(trial.last_checkpoint_location.worker_id)

    def _trial_status_to_str(self, status: TrialStatus) -> str:
        reset = "\033[0m"
        red = "\033[91m"
        green = "\033[92m"
        yellow = "\033[93m"
        blue = "\033[94m"

        match status:
            case TrialStatus.RUNNING:
                return f"{green}{status:^11}{reset}"
            case TrialStatus.PENDING:
                return f"{blue}{status:^11}{reset}"
            case TrialStatus.WAITING:
                return f"{yellow}{status:^11}{reset}"
            case TrialStatus.TERMINATED:
                return f"{red}{status:^11}{reset}"
            case TrialStatus.FAILED:
                return f"{status}"

    def _worker_ip_to_str(self, worker_id: int) -> str:
        return self.worker_states[worker_id].node_name.split(".")[-1]

    def display_trial_result(
        self,
        output_path: Path = TRIAL_PROGRESS_OUTPUT_PATH,
    ) -> None:
        try:
            with output_path.open("w") as f:
                # 表頭
                headers = [
                    "ID",
                    "Status",
                    "SaveAt",
                    "IP",
                    "WID",
                    "Type",
                    "Hyperparameter",
                    "Gene",
                    "Acc",
                ]
                widths = [4, 11, 6, 4, 4, 6, 60, 7, 7]  # Hyperparameter 欄位加寬

                # ┏━┳━┓
                f.write("┏" + "┳".join("━" * w for w in widths) + "┓\n")
                # 標題列
                f.write(
                    "┃" + "┃".join(h.center(w) for h, w in zip(headers, widths)) + "┃\n"
                )
                # 分隔線
                f.write("┣" + "╋".join("━" * w for w in widths) + "┫\n")

                # trial 列
                for i in self.all_trials.values():
                    worker_type = self._worker_type_to_str(i.worker_type)
                    worker_id = self._worker_id_to_str(i.worker_id)
                    save_at = self._save_at_to_str(i)
                    status = self._trial_status_to_str(i.status)
                    ip = (
                        self._worker_ip_to_str(i.worker_id)
                        if i.worker_id != -1
                        else "-"
                    )

                    h = i.hyperparameter
                    model_name = getattr(h, "model_name", "-")
                    momentum = getattr(h, "momentum", 0.0)

                    hyperparam_str = str(h)

                    row = [
                        str(i.id),
                        status,
                        save_at,
                        ip,
                        worker_id,
                        worker_type,
                        hyperparam_str,
                        str(i.generation),
                        f"{i.accuracy:.3f}",
                    ]
                    f.write(
                        "┃"
                        + "┃".join(val.rjust(w) for val, w in zip(row, widths))
                        + "┃\n"
                    )

                # 底線
                f.write("┗" + "┻".join("━" * w for w in widths) + "┛\n")
                timestamp = (datetime.now(UTC) + timedelta(hours=8)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                f.write(f"{timestamp}\n")

        except Exception as e:
            print(f"Error writing trial results: {e}")

    def print_iteration_count(self) -> None:
        iteration_counts = [
            (i.id, i.device_iteration_count) for i in self.all_trials.values()
        ]

        iteration_counts.sort(key=lambda x: x[0])

        for index, value in iteration_counts:
            self.logger.info(
                "Trial:%2d CPU/GPU %s",
                index,
                colored_progress_bar(
                    [value[WorkerType.CPU], value[WorkerType.GPU]],
                    40,
                ),
            )
        self.logger.info(
            "Total    CPU/GPU %s",
            colored_progress_bar(
                [
                    sum(i[1][WorkerType.CPU] for i in iteration_counts),
                    sum(i[1][WorkerType.GPU] for i in iteration_counts),
                ],
                40,
            ),
        )
