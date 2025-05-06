import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Any, List, Optional, Protocol

import ray
from ray import ObjectRef
from ray.actor import ActorHandle

from .config import PHASE_ITERATION, STOP_ITERATION
from .trial_phase import TrialPhase
from .trial_state import TrialState
from .utils import TrialStatus, WorkerType, colored_progress_bar


class AssignTrialStrategy(Protocol):
    def __call__(
        self,
        trial_state: List[TrialState],
        gpu_workers: List[ActorHandle],
        cpu_workers: List[ActorHandle],
        *args: Any,
        **kwargs: Any,
    ) -> List[ObjectRef]: ...


def round_robin_strategy(
    pending_trial_states: List[TrialState],
    gpu_workers: List[ActorHandle],
    cpu_workers: List[ActorHandle],
    trial_phase: TrialPhase,
) -> Optional[ObjectRef]:

    if not pending_trial_states:
        return None

    # Assign to CPU
    available_futures = [worker.get_available_slots.remote() for worker in cpu_workers]

    available_cpu_workers = [
        worker
        for worker, available_slots in zip(cpu_workers, ray.get(available_futures))  # type: ignore
        if available_slots
    ]

    if available_cpu_workers:
        worker = next(iter(available_cpu_workers))
        available_trials = [
            trial
            for trial in pending_trial_states
            if trial.phase <= trial_phase.current_phase
        ]

        if available_trials:
            trial_state = max(
                available_trials,
                key=lambda x: x.iteration,
            )

            pending_trial_states.remove(trial_state)
            future = worker.assign_trial.remote(trial_state)

            return future

    # Assign to GPU
    available_futures = [worker.get_available_slots.remote() for worker in gpu_workers]

    available_gpu_workers = [
        (worker, available_slots)
        for worker, available_slots in zip(gpu_workers, ray.get(available_futures))  # type: ignore
        if available_slots
    ]

    if available_gpu_workers:
        worker = max(available_gpu_workers, key=lambda x: x[1])[0]
        trial_state = min(pending_trial_states, key=lambda x: x.iteration)

        pending_trial_states.remove(trial_state)
        future = worker.assign_trial.remote(trial_state)

        return future


def gpu_first_strategy(
    pending_trial_states: List[TrialState],
    gpu_workers: List[ActorHandle],
    cpu_workers: List[ActorHandle],
    **kargs: Any,
) -> Optional[ObjectRef]:
    logger = kargs["logger"]

    available_futures = [worker.get_available_slots.remote() for worker in gpu_workers]
    available_gpu_workers = [
        (worker, available_slots)
        for worker, available_slots in zip(gpu_workers, ray.get(available_futures))  # type: ignore
        if available_slots
    ]

    if not available_gpu_workers:
        return None

    if len(pending_trial_states):
        worker = max(available_gpu_workers, key=lambda x: x[1])[0]
        trial_state = min(pending_trial_states, key=lambda x: x.iteration)

        pending_trial_states.remove(trial_state)
        future = worker.assign_trial.remote(trial_state)

        return future

    available_futures = [worker.get_active_trials.remote() for worker in cpu_workers]

    running_cpu_workers = [
        (worker, min(activate_trials, key=lambda x: x.iteration))
        for worker, activate_trials in zip(cpu_workers, ray.get(available_futures))  # type: ignore
        if len(activate_trials) > 0
    ]

    if running_cpu_workers:
        worker, trial_state = min(running_cpu_workers, key=lambda x: x[1].iteration)
        logger.info(f"對 Trial {trial_state.id} 執行搶奪")
        ray.get(worker.send_signal.remote(trial_state.id))  # type: ignore


def get_trial_scheduler_logger() -> logging.Logger:
    """
    設置並返回一個日誌記錄器，用於跟踪訓練過程中的 TrialScheduler 記錄。

    日誌將記錄到一個帶有時間戳的目錄中，並包括在終端顯示和日誌文件中的訊息。

    Returns:
        logging.Logger: 配置好的 TrialScheduler 記錄器。
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(os.getcwd(), "logs/", timestamp)
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(f"trial_scheduler")

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # 或者選擇更合適的級別

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s TRIAL_SCHEDULER -- %(message)s",
            # datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)  # 只顯示 INFO 級別以上的訊息
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"trial_scheduler.log")
        )
        file_handler.setLevel(logging.DEBUG)  # 記錄所有級別的日誌
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class TrialScheduler:
    """
    試驗調度器，負責管理和分配訓練試驗給可用的工作者。

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
        tuner: ActorHandle,
        workers: List[ActorHandle],
        trial_states: List[TrialState],
    ) -> None:
        """
        初始化 TrialScheduler，設置試驗狀態和工作者。

        Args:
            train_step (TrainStepFunction): 訓練步驟函數。
            trial_states (List[TrialState]): 初始的試驗狀態列表。
        """
        self.tuner = tuner
        self.trial_phase = TrialPhase(STOP_ITERATION, PHASE_ITERATION)

        self.pending_trial_states: List[TrialState] = trial_states
        self.completed_trial_states: List[TrialState] = []
        self.waiting_trial_states: List[TrialState] = []
        self.trial_state_nums = len(self.pending_trial_states)

        self.running_futures: List[ObjectRef] = []
        self.logger = get_trial_scheduler_logger()
        self.workers = workers
        self._previous_time = time.time()

        self.gpu_workers = [
            worker
            for worker in self.workers
            if ray.get(worker.get_worker_type.remote()) == WorkerType.GPU  # type: ignore
        ]
        self.idle_gpu_count = 0

        self.cpu_workers = [
            worker
            for worker in self.workers
            if ray.get(worker.get_worker_type.remote()) == WorkerType.CPU  # type:ignore
        ]

        self.logger.debug(f"{len(self.gpu_workers)=}")
        self.logger.debug(f"{len(self.cpu_workers)=}")

    def assign_trial_to_worker(self) -> None:  # type: ignore
        """
        將一個試驗分配給一個可用的工作者。

        如果所有工作者都忙碌，則返回當前正在運行的訓練任務。

        Returns:
            List[ObjectRef]: 當前正在運行的訓練任務列表。
        """
        self.update_phase()

        if self.pending_trial_states:
            pending_trial_list = sorted(
                self.pending_trial_states, key=lambda t: t.iteration
            )
            pending_trial_id_list = [i.id for i in pending_trial_list]
            self.logger.info(
                f"⏳ 等待中訓練任務列表長度：{len(pending_trial_list):2d}, {pending_trial_id_list}"
            )

        future: Optional[ObjectRef] = None
        if (
            len(self.completed_trial_states)
            <= self.trial_state_nums - len(self.gpu_workers) * 3
        ):
            future = round_robin_strategy(
                pending_trial_states=self.pending_trial_states,
                gpu_workers=self.gpu_workers,
                cpu_workers=self.cpu_workers,
                trial_phase=self.trial_phase,
            )
        else:
            gpu_first_strategy(
                self.pending_trial_states,
                self.gpu_workers,
                self.cpu_workers,
                logger=self.logger,
            )

        if future is not None:
            self.running_futures.append(future)

    def run(self):
        """
        開始訓練過程，將試驗分配給工作者並處理完成的結果。

        該方法會持續運行直到所有的試驗都完成。
        """
        self.logger.info("訓練開始")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # while self.running_futures or self.pending_trial_states:
        while len(self.completed_trial_states) < self.trial_state_nums:
            self.assign_trial_to_worker()

            if not self.running_futures and not self.pending_trial_states:
                break

            done_futures, self.running_futures = ray.wait(
                self.running_futures, timeout=0.5
            )

            if done_futures:
                loop.run_until_complete(self.handle_done_futures(done_futures))

        self.print_iteration_count()
        self.logger.info("🎉 所有 Trial 訓練完成！")

    def print_iteration_count(self) -> None:
        iteration_counts = [
            (i.id, i.device_iteration_count) for i in self.completed_trial_states
        ]

        iteration_counts.sort(key=lambda x: x[0])

        for index, value in iteration_counts:
            print(
                f"Trial:{index:2} CPU/GPU",
                colored_progress_bar(
                    [value[WorkerType.CPU], value[WorkerType.GPU]], 40
                ),
            )

        print(
            f"Total   CPU/GPU",
            colored_progress_bar(
                [
                    sum(i[1][WorkerType.CPU] for i in iteration_counts),
                    sum(i[1][WorkerType.GPU] for i in iteration_counts),
                ],
                40,
            ),
        )

    def get_workers_logs(self) -> None:
        """
        獲取所有工作者的日誌並將其保存到文件中。

        該方法會將每個工作者的日誌寫入到相應的文件中。
        """

        log_dir = None

        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_dir = os.path.dirname(handler.baseFilename)  # 取得資料夾路徑
                break

        if log_dir is None:
            self.logger.error("logs檔案資料夾不存在")
            return

        for worker in self.workers:
            future = ray.get(worker.get_log_file.remote())  # type: ignore
            with open(os.path.join(log_dir, f"worker-{future['id']}.log"), "w") as f:
                f.write(future["content"])

    def update_phase(self):
        old = self.trial_phase.current_phase

        self.trial_phase.update_phase(ray.get(self.tuner.get_trial_progress.remote()))  # type: ignore

        if old != self.trial_phase.current_phase:
            self.logger.info(f"更新階段到Phase {self.trial_phase.current_phase}")
            futures = [
                worker.update_phase.remote(self.trial_phase.current_phase)
                for worker in self.workers
            ]
            ray.get(futures)  # type: ignore

    async def handle_done_futures(self, done_futures: List[ObjectRef]):
        """
        處理已完成的訓練任務，將結果添加到已完成試驗狀態列表中。

        Args:
            done_futures (List[ObjectRef]): 已完成的訓練任務列表。
        """

        for future in done_futures:
            try:
                trial_state: TrialState = ray.get(future)
                if trial_state.status == TrialStatus.TERMINATE:
                    self.completed_trial_states.append(trial_state)
                    self.logger.info(
                        f"✅ Worker {trial_state.worker_id} Trial {trial_state.id} 完成，Accuracy: {trial_state.accuracy:.1f}"
                    )
                    self.logger.info(
                        f"✅ 已完成的訓練任務列表: {sorted([i.id for i in self.completed_trial_states])}"
                    )

                elif trial_state.status == TrialStatus.NEED_MUTATION:
                    trial_state = ray.get(self.tuner.mutation.remote(trial_state))  # type: ignore
                    trial_state.status = TrialStatus.PENDING
                    self.pending_trial_states.append(trial_state)

                elif trial_state.status == TrialStatus.PAUSE:
                    trial_state.status = TrialStatus.PENDING
                    self.pending_trial_states.append(trial_state)
                    self.logger.info(
                        f"🔃 Worker {trial_state.worker_id} 回傳未完成 Trial {trial_state.id}, Iteration: {trial_state.iteration} ，Accuracy: {trial_state.accuracy:.2f}"
                    )

                elif trial_state.status == TrialStatus.PENDING:
                    self.pending_trial_states.append(trial_state)
                    self.logger.warning(f"❗發生碰撞, 回傳 Trial {trial_state.id}")

                trial_state.worker_id = -1
                trial_state.worker_type = None
                self.tuner.record_trial_progress.remote(trial_state)
            except Exception as e:
                self.logger.error(f"❌ Future 執行失敗: {e}")
