import argparse
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from itertools import islice
from pathlib import Path

import ray
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

from src.config import (
    DATASET_PATH,
    ITERATION_PER_GENERATION,
    MAX_GENERATION,
    STOP_ACCURACY,
)
from src.trial_state import TrialState
from src.tuner import Tuner
from src.utils import Checkpoint, Hyperparameter, get_head_node_address, unzip_file

DEFAULT_DEVICE = torch.device("cpu")


def cifar100_data_loader_factory(
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_dir = Path(DATASET_PATH).expanduser()
    if not (Path(data_dir) / "cifar-100-python").exists():
        print(f"{data_dir} 不存在")
        Path(data_dir).mkdir(parents=True, exist_ok=True)

    if not (Path(data_dir) / "cifar-100-python").exists():
        print(f"{Path(data_dir) / 'cifar-100-python'} 不存在")

        torchvision.datasets.CIFAR100(
            root=data_dir,
            train=True,
            download=True,
            transform=None,
        )
        print(f"Dataset downloaded to {data_dir}")

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ],
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ],
    )

    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=False,
        transform=train_transform,
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=False,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, valid_loader, None


def cifar10_data_loader_factory(
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_dir = Path(DATASET_PATH).expanduser()
    if not Path(data_dir).exists():
        Path(data_dir).mkdir(parents=True, exist_ok=True)

    if not (Path(data_dir) / "cifar-10-batches-py").exists():
        print(f"{Path(data_dir) / 'cifar-10-batches-py'} 不存在")
        torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=None,
        )

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ],
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ],
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=train_transform,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, valid_loader, None


def resnet18_init_fn(
    hyperparameter: Hyperparameter,
    checkpoint: Checkpoint,
    device: torch.device,
) -> tuple[nn.Module, optim.Optimizer]:
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 10)

    if checkpoint.is_empty():
        model.to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=hyperparameter.lr,
            momentum=hyperparameter.momentum,
        )

        return model, optimizer

    model.load_state_dict(checkpoint.model_state_dict)
    model = model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=hyperparameter.lr,
        momentum=hyperparameter.momentum,
    )
    optimizer.load_state_dict(checkpoint.optimizer_state_dict)

    for param_group in optimizer.param_groups:
        param_group["lr"] = hyperparameter.lr
        param_group["momentum"] = hyperparameter.momentum

    return model, optimizer


def resnet50_init_fn(
    hyperparameter: Hyperparameter,
    checkpoint: Checkpoint,
    device: torch.device,
) -> tuple[nn.Module, optim.Optimizer]:
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 100)

    if checkpoint.is_empty():
        model.to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=hyperparameter.lr,
            momentum=hyperparameter.momentum,
        )

        return model, optimizer

    model.load_state_dict(checkpoint.model_state_dict)
    model = model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=hyperparameter.lr,
        momentum=hyperparameter.momentum,
    )
    optimizer.load_state_dict(checkpoint.optimizer_state_dict)

    for param_group in optimizer.param_groups:
        param_group["lr"] = hyperparameter.lr
        param_group["momentum"] = hyperparameter.momentum

    return model, optimizer


def generate_trial_states(
    n: int = 1,
    model_init_fn: Callable = resnet18_init_fn,
) -> list[TrialState]:
    return [
        TrialState(
            i,
            Hyperparameter.random(),
            model_init_fn,
        )
        for i in range(n)
    ]


def train_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    batch_size: int,
    device: torch.device = DEFAULT_DEVICE,
) -> None:
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)

    for raw_inputs, raw_targets in islice(train_loader, 1):
        inputs, targets = raw_inputs.to(device), raw_targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_num", type=int, default=40)
    parser.add_argument(
        "--model",
        choices=["resnet-18", "resnet-50"],
        default="resnet-18",
    )
    parser.add_argument(
        "--dataset",
        choices=["cifar-10", "cifar-100"],
        default="cifar-10",
    )

    args = parser.parse_args()

    trial_num = args.trial_num
    model_name = args.model
    dataset = args.dataset
    print(trial_num, model_name, dataset)

    print(f"STOP_ACCURACY {STOP_ACCURACY}")
    print(f"MAX_GENERATION {MAX_GENERATION}")
    print(f"ITERATION_PER_GENERATION {ITERATION_PER_GENERATION}")

    model_fn = resnet18_init_fn
    output_dir = "ResNet50"
    match model_name:
        case "resnet-18":
            model_fn = resnet18_init_fn
            output_dir = "ResNet18"
        case "resnet-50":
            model_fn = resnet50_init_fn
            output_dir = "ResNet50"

    dataset_fn = cifar10_data_loader_factory
    match dataset:
        case "cifar-10":
            dataset_fn = cifar10_data_loader_factory
        case "cifar-100":
            dataset_fn = cifar100_data_loader_factory

    ray.init(
        runtime_env={
            "working_dir": ".",
            "excludes": [
                ".git",
                "test",
                "logs/*",
                "LICENSE",
                "README.md",
                ".venv",
                ".ruff_cache",
            ],
        },
    )
    trial_states = generate_trial_states(trial_num, model_fn)

    bs_list = [32, 64, 128, 256, 512]
    for i in range(len(trial_states)):
        batch_size = bs_list[i % len(bs_list)]
        trial_states[i].hyperparameter.batch_size = batch_size

    tuner = Tuner.options(  # type: ignore[call-arg]
        max_concurrency=5,
        num_cpus=1,
        resources={f"node:{get_head_node_address()}": 0.01},
    ).remote(trial_states, train_step, dataset_fn)
    ray.get(tuner.run.remote())  # type: ignore[call-arg]

    zip_logs_bytes: bytes = ray.get(tuner.get_zipped_log.remote())  # type: ignore[call-arg]

    time_stamp = (datetime.now(UTC) + timedelta(hours=8)).strftime("%Y-%m-%d_%H-%M-%S")
    zip_output_dir = Path("./logs") / output_dir / f"Trial{trial_num}" / time_stamp

    zip_output_dir.mkdir(parents=True, exist_ok=True)
    zip_output_path = Path(zip_output_dir) / "logs.zip"
    with Path(zip_output_path).open("wb") as f:
        f.write(zip_logs_bytes)

    unzip_file(zip_output_path, zip_output_dir)  # type: ignore[call-arg]

    ray.shutdown()
