import inspect
import logging
import random
import time
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from functools import reduce, wraps
from typing import Any, ParamSpec, TypeVar

import psutil
import ray
import torch
from ray.actor import ActorHandle

# ╭──────────────────────────────────────────────────────────╮
# │                          Enums                           │
# ╰──────────────────────────────────────────────────────────╯


class ModelType(Enum):
    RESNET_18 = auto()
    RESNET_50 = auto()

    def __str__(self) -> str:
        return self.name


class TrialStatus(Enum):
    RUNNING = auto()
    PENDING = auto()
    TERMINATED = auto()
    FAILED = auto()
    WAITING = auto()

    def __str__(self) -> str:
        return self.name


class WorkerType(Enum):
    CPU = auto()
    GPU = auto()


class DatasetType(Enum):
    CIFAR10 = auto()
    CIFAR100 = auto()
    IMAGENET = auto()


# ╭──────────────────────────────────────────────────────────╮
# │                       Dataclasses                        │
# ╰──────────────────────────────────────────────────────────╯


@dataclass
class WorkerState:
    id: int
    num_cpus: int
    num_gpus: int
    node_name: str
    max_trials: int = 1
    worker_type: WorkerType = WorkerType.CPU

    total_train_time: float = 0.0
    train_step_count: int = 0

    @property
    def avg_train_time(self) -> float:
        if self.train_step_count == 0:
            return 0.0
        return self.total_train_time / self.train_step_count

    def record_train_time(self, duration: float) -> None:
        self.total_train_time += duration
        self.train_step_count += 1


@dataclass
class Hyperparameter:
    lr: float
    batch_size: int

    @classmethod
    def random(cls) -> "Hyperparameter":
        return cls(
            lr=random.uniform(0.001, 0.1),
            batch_size=random.choice([32, 64, 128, 512, 1024]),
        )

    def explore(self) -> "Hyperparameter":
        return Hyperparameter(
            self.lr * 0.8,
            self.batch_size,
        )


@dataclass
class Checkpoint:
    model_state_dict: dict
    optimizer_state_dict: dict

    @classmethod
    def empty(cls) -> "Checkpoint":
        return cls(model_state_dict={}, optimizer_state_dict={})

    def is_empty(self) -> bool:
        return not self.model_state_dict and not self.optimizer_state_dict


@dataclass
class CheckpointLocation:
    worker_id: int | None
    worker_reference: ActorHandle | None

    @classmethod
    def empty(cls) -> "CheckpointLocation":
        return cls(worker_id=None, worker_reference=None)

    def is_empty(self) -> bool:
        return self.worker_id is None and self.worker_reference is None


# ╭──────────────────────────────────────────────────────────╮
# │                       Type Define                        │
# ╰──────────────────────────────────────────────────────────╯


P = ParamSpec("P")
R = TypeVar("R")


# ╭──────────────────────────────────────────────────────────╮
# │                        Functions                         │
# ╰──────────────────────────────────────────────────────────╯

T = TypeVar("T")
Composeable = Callable[[T], T]


def compose(*functions: Composeable) -> Composeable:
    def apply(value: T, fn: Composeable[T]) -> T:
        return fn(value)

    return lambda data: reduce(apply, functions[::-1], data)


def pipe(*functions: Composeable) -> Composeable:
    def apply(value: T, fn: Composeable[T]) -> T:
        return fn(value)

    return lambda data: reduce(apply, functions, data)


def get_head_node_address() -> str:
    return ray.get_runtime_context().gcs_address.split(":")[0]


def colored_progress_bar(data: list[int], bar_width: int) -> str:
    green = "\033[92m"
    red = "\033[91m"
    yellow = "\033[93m"
    reset = "\033[0m"

    colors = [green, red, yellow]
    total = sum(data)
    if total == 0:
        return " " * bar_width + " (no data)"

    percentages = [x / total for x in data]
    lengths = [int(p * bar_width) for p in percentages]

    while sum(lengths) < bar_width:
        max_idx = percentages.index(max(percentages))
        lengths[max_idx] += 1

    bar = "".join(
        colors[i % len(colors)] + "━" * length for i, length in enumerate(lengths)
    )
    bar += reset

    data_str = "/".join([f"{x:04d}" for x in data])
    perc_str = "/".join([f"{p * 100:.2f}%" for p in percentages])

    return f"{bar}  {data_str}  {perc_str}"


def unzip_file(zip_path: str, extract_to: str) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


@contextmanager
def timing_block(
    label: str,
    logger: Callable[[str], None] | None = None,
) -> Iterator[None]:
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    msg = f"{label} 花費 {end - start:.6f} 秒"
    if logger:
        logger(msg)
    else:
        print(msg)  # noqa: T201


def get_tensor_dict_size(state_dict: dict) -> int:
    total = 0
    for v in state_dict.values():
        if isinstance(v, dict):
            total += get_tensor_dict_size(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    total += get_tensor_dict_size(item)
        elif hasattr(v, "numel") and hasattr(v, "element_size"):
            total += v.numel() * v.element_size()
    return total


# ╭──────────────────────────────────────────────────────────╮
# │                        Decorators                        │
# ╰──────────────────────────────────────────────────────────╯


def timer() -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            message = f"Function '{func.__name__}' 花費 {end - start:.6f} 秒"

            if args and hasattr(args[0], "logger"):
                logger = getattr(args[0], "logger", None)

                if isinstance(logger, logging.Logger | logging.LoggerAdapter):
                    logger.info(message)
                else:
                    print(message)  # noqa: T201

            else:
                print(message)  # noqa: T201
            return result

        return wrapper

    return decorator


def resource_monitor() -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            self = args[0]
            result = func(*args, **kwargs)

            tb_manager = getattr(self, "tb_manager", None)
            worker_state = getattr(self, "worker_state", None)

            if tb_manager and worker_state:
                cpu_usage = psutil.cpu_percent()
                mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)

                gpu_usage = 0.0
                gpu_mem = 0.0
                if torch.cuda.is_available():
                    # 取得目前 GPU 的記憶體使用量 (MB)
                    gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)
                    # 註：GPU 利用率 % 通常需要 pynvml，簡單實作可先記為 0 或僅紀錄記憶體

                # 異步傳送數據到 TensorBoard
                self.tb_manager.add_resource_usage.remote(
                    worker_id=self.worker_state.id,
                    cpu=cpu_usage,
                    mem=mem_mb,
                    gpu=gpu_usage,
                    gpu_mem=gpu_mem,
                    timestamp=time.time(),
                )

            return result

        return wrapper

    return decorator


def rc(func: Callable[P, R]) -> Callable[P, R]:
    signature = inspect.signature(func)

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()

        self_obj = bound.arguments["self"]
        trial_id = bound.arguments["trial_id"]
        fn_anme = func.__name__

        try:
            self_obj.rc[trial_id] += 1
            self_obj.logger.debug(
                "Enter `%s` trial_id=%d, rc=%d",
                fn_anme,
                trial_id,
                self_obj.rc[trial_id],
            )
            return func(*bound.args, **bound.kwargs)

        finally:
            self_obj.rc[trial_id] -= 1
            self_obj.logger.debug(
                "Exit  `%s` trial_id=%d, rc=%d",
                fn_anme,
                trial_id,
                self_obj.rc[trial_id],
            )

    return wrapper
