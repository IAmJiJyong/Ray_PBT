from itertools import islice
from pathlib import Path
from typing import Protocol, TypeVar, cast

import torch
from datasets import load_dataset, logging
from torch import nn, no_grad, optim
from torch.types import Device
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import CIFAR10, CIFAR100
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    MobileBertForSequenceClassification,
)

from .config import DATASET_PATH
from .hyperparameter import BertHyperparameter, CNNHyperparameter, Hyperparameter
from .utils import Checkpoint

H_contra = TypeVar("H_contra", bound=Hyperparameter, contravariant=True)

logging.disable_progress_bar()
logging.set_verbosity_error()


class TaskStrategy(Protocol[H_contra]):
    def build_model(
        self,
        hyperparameter: H_contra,
        checkpoint: Checkpoint,
        device: Device,
    ) -> nn.Module: ...

    def build_optimizer(
        self,
        model: nn.Module,
        hyperparameter: H_contra,
        checkpoint: Checkpoint,
        device: Device,
    ) -> optim.Optimizer: ...

    def build_dataloaders(
        self,
        hyperparameter: H_contra,
    ) -> tuple[DataLoader, DataLoader, DataLoader]: ...

    def train_step(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        dataloader: DataLoader,
        device: Device,
    ) -> None: ...

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: Device,
    ) -> float: ...


class BertSST2Task(TaskStrategy[BertHyperparameter]):
    def build_model(
        self,
        hyperparameter: BertHyperparameter,
        checkpoint: Checkpoint,
        device: Device,
    ) -> nn.Module:
        model = MobileBertForSequenceClassification.from_pretrained(
            "google/mobilebert-uncased",
            num_labels=2,
        )

        if checkpoint.is_empty():
            model.to(device)
            return model

        model.load_state_dict(checkpoint.model_state_dict)
        model.to(device)

        return model

    def build_optimizer(
        self,
        model: nn.Module,
        hyperparameter: BertHyperparameter,
        checkpoint: Checkpoint,
        device: Device,
    ) -> optim.Optimizer:
        hyperparameter = cast("BertHyperparameter", hyperparameter)

        if checkpoint.is_empty():
            return optim.AdamW(
                model.parameters(),
                lr=hyperparameter.lr,
                weight_decay=hyperparameter.weight_decay,
            )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=hyperparameter.lr,
            weight_decay=hyperparameter.weight_decay,
        )
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)

        for param_group in optimizer.param_groups:
            param_group["lr"] = hyperparameter.lr
            param_group["weight_decay"] = hyperparameter.weight_decay

        return optimizer

    def build_dataloaders(
        self,
        hyperparameter: BertHyperparameter,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        data_dir: Path = Path(
            "~/Documents/workspace/tune_population_based/data",
        ).expanduser()
        batch_size = hyperparameter.batch_size

        tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        dataset = load_dataset("glue", "sst2", cache_dir=str(data_dir))

        tokenized_datasets = dataset.map(
            lambda exam: tokenizer(
                exam["sentence"],
                truncation=True,
                max_length=128,
            ),
            batched=True,
        )
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        train_loader = DataLoader(
            tokenized_datasets["train"],
            shuffle=True,
            batch_size=batch_size,
            collate_fn=data_collator,
            pin_memory=True,  # 針對 GPU 加速資料搬運
        )

        test_loader = DataLoader(
            tokenized_datasets["validation"],
            batch_size=batch_size,
            collate_fn=data_collator,
            pin_memory=True,  # 針對 GPU 加速資料搬運
        )

        return train_loader, test_loader, None

    def train_step(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        dataloader: DataLoader,
        device: Device,
    ) -> None:
        model.train()
        # for raw_batch in islice(dataloader, 50):
        for raw_batch in dataloader:
            batch = {k: v.to(device) for k, v in raw_batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    @no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: Device,
    ) -> float:
        model.eval()

        correct = 0
        total = 0

        for raw_batch in dataloader:
            batch = {k: v.to(device) for k, v in raw_batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)

            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

        return correct / total


class ResNet18Cifar10Task(TaskStrategy[CNNHyperparameter]):
    def build_model(
        self,
        hyperparameter: CNNHyperparameter,
        checkpoint: Checkpoint,
        device: Device,
    ) -> nn.Module:
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, 10)

        if checkpoint.is_empty():
            model.to(device)
            return model

        model.load_state_dict(checkpoint.model_state_dict)
        model.to(device)
        return model

    def build_optimizer(
        self,
        model: nn.Module,
        hyperparameter: CNNHyperparameter,
        checkpoint: Checkpoint,
        device: Device,
    ) -> optim.Optimizer:
        if checkpoint.is_empty():
            return optim.SGD(
                model.parameters(),
                lr=hyperparameter.lr,
                momentum=hyperparameter.momentum,
            )

        optimizer = optim.SGD(
            model.parameters(),
            lr=hyperparameter.lr,
            momentum=hyperparameter.momentum,
        )
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)

        for param_group in optimizer.param_groups:
            param_group["lr"] = hyperparameter.lr
            param_group["momentum"] = hyperparameter.momentum

        return optimizer

    def build_dataloaders(
        self,
        hyperparameter: CNNHyperparameter,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        data_dir = Path(DATASET_PATH).expanduser()
        if not Path(data_dir).exists():
            Path(data_dir).mkdir(parents=True, exist_ok=True)

        if not (Path(data_dir) / "cifar-10-batches-py").exists():
            print(f"{Path(data_dir) / 'cifar-10-batches-py'} 不存在")
            CIFAR10(
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
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ],
        )

        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ],
        )

        train_dataset = CIFAR10(
            root=data_dir,
            train=True,
            download=False,
            transform=train_transform,
        )
        test_dataset = CIFAR10(
            root=data_dir,
            train=False,
            download=False,
            transform=test_transform,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=hyperparameter.batch_size,
            shuffle=True,
        )
        valid_loader = DataLoader(
            test_dataset,
            batch_size=hyperparameter.batch_size,
            shuffle=False,
        )

        return train_loader, valid_loader, None

    def train_step(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        dataloader: DataLoader,
        device: Device,
    ) -> None:
        model.train()
        criterion = nn.CrossEntropyLoss().to(device)

        for raw_inputs, raw_targets in islice(dataloader, 1):
            inputs, targets = raw_inputs.to(device), raw_targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: Device,
    ) -> float:
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for raw_inputs, raw_targets in dataloader:
                inputs, targets = (
                    raw_inputs.to(device),
                    raw_targets.to(device),
                )
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return correct / total


class ResNet50CIFAR100Task(TaskStrategy[CNNHyperparameter]):
    def build_model(
        self,
        hyperparameter: CNNHyperparameter,
        checkpoint: Checkpoint,
        device: Device,
    ) -> nn.Module:
        model = models.resnet50(num_classes=100)
        model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.maxpool = nn.Identity()

        if checkpoint.is_empty():
            model.to(device)
            return model

        model.load_state_dict(checkpoint.model_state_dict)
        model.to(device)

        return model

    def build_optimizer(
        self,
        model: nn.Module,
        hyperparameter: CNNHyperparameter,
        checkpoint: Checkpoint,
        device: Device,
    ) -> optim.Optimizer:
        if checkpoint.is_empty():
            return optim.SGD(
                model.parameters(),
                lr=hyperparameter.lr,
                momentum=hyperparameter.momentum,
            )

        optimizer = optim.SGD(
            model.parameters(),
            lr=hyperparameter.lr,
            momentum=hyperparameter.momentum,
        )
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)

        for param_group in optimizer.param_groups:
            param_group["lr"] = hyperparameter.lr
            param_group["momentum"] = hyperparameter.momentum

        return optimizer

    def build_dataloaders(
        self,
        hyperparameter: CNNHyperparameter,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        data_dir = Path(DATASET_PATH).expanduser()
        if not (Path(data_dir) / "cifar-100-python").exists():
            print(f"{data_dir} 不存在")
            Path(data_dir).mkdir(parents=True, exist_ok=True)

        if not (Path(data_dir) / "cifar-100-python").exists():
            print(f"{Path(data_dir) / 'cifar-100-python'} 不存在")

            CIFAR100(
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

        train_dataset = CIFAR100(
            root=data_dir,
            train=True,
            download=False,
            transform=train_transform,
        )
        test_dataset = CIFAR100(
            root=data_dir,
            train=False,
            download=False,
            transform=test_transform,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=hyperparameter.batch_size,
            shuffle=True,
        )

        valid_loader = DataLoader(
            test_dataset,
            batch_size=hyperparameter.batch_size,
            shuffle=False,
        )

        return train_loader, valid_loader, None

    def train_step(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        dataloader: DataLoader,
        device: Device,
    ) -> None:
        model.train()
        criterion = nn.CrossEntropyLoss().to(device)

        for raw_inputs, raw_targets in islice(dataloader, 30):
            inputs, targets = raw_inputs.to(device), raw_targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: Device,
    ) -> float:
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for raw_inputs, raw_targets in dataloader:
                inputs, targets = (
                    raw_inputs.to(device),
                    raw_targets.to(device),
                )
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return correct / total
