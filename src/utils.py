import logging
import random
import time
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from functools import reduce, wraps
from itertools import repeat
from typing import ParamSpec, Protocol, TypeVar

import numpy as np
import ray
from numpy._typing import NDArray
from ray.actor import ActorHandle
from torch import device, nn, optim
from torch.utils.data import DataLoader

from .config import MAX_BATCH_SIZE, MIN_BATCH_SIZE, PERTURBATION_FACTOR

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
    momentum: float
    weight_decay: float
    dampening: float
    batch_size: int
    model_type: ModelType

    @staticmethod
    def to_batch_size(x: float) -> int:
        return int(x * float(MAX_BATCH_SIZE - MIN_BATCH_SIZE)) + MIN_BATCH_SIZE

    @staticmethod
    def batch_size_to_float(x: int) -> float:
        return (x - MIN_BATCH_SIZE) / float(MAX_BATCH_SIZE - MIN_BATCH_SIZE)

    def __str__(self) -> str:
        return (
            f"Hyperparameter(lr:{self.lr:.3f}, momentum:{self.momentum:.3f}, "
            f"batch_size:{self.batch_size:4d}, model_type:{self.model_type})"
        )

    @classmethod
    def random(cls) -> "Hyperparameter":
        return cls(
            lr=random.uniform(0.001, 1),
            momentum=random.uniform(0.001, 1),
            weight_decay=random.uniform(1e-7, 1e-4),
            dampening=random.uniform(1e-7, 1e-4),
            batch_size=Hyperparameter.to_batch_size(random.uniform(0, 1)),
            model_type=ModelType.RESNET_18,
        )

    def to_ndarray(self) -> NDArray[np.floating]:
        return np.array(
            [
                self.lr,
                self.momentum,
                self.weight_decay,
                self.dampening,
                Hyperparameter.batch_size_to_float(self.batch_size),
            ],
        )

    def explore(self) -> "Hyperparameter":
        hyper = self.to_ndarray()
        perturb = np.array(
            random.choices(
                [1.0 - PERTURBATION_FACTOR, 1.0 + PERTURBATION_FACTOR, 1.0, 1.0],
            )
            for _ in repeat(None, 5)
        )
        new_hyper = hyper * perturb

        return Hyperparameter(
            lr=new_hyper[0],
            momentum=new_hyper[1],
            weight_decay=new_hyper[2],
            dampening=new_hyper[3],
            batch_size=Hyperparameter.to_batch_size(new_hyper[4]),
            model_type=self.model_type,
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


class TrainStepFunction(Protocol[P]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None: ...


class DataloaderFactory(Protocol):
    def __call__(
        self,
        batch_size: int,
        num_workers: int,
    ) -> tuple[DataLoader, DataLoader, DataLoader]: ...


class ModelInitFunction(Protocol):
    def __call__(
        self,
        hyperparameter: Hyperparameter,
        checkpoint: Checkpoint,
        device: device,
    ) -> tuple[nn.Module, optim.Optimizer]: ...


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
