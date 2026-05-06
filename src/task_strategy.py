from itertools import islice
from pathlib import Path
from random import shuffle
from typing import Any, Protocol, TypeVar, cast

import torch
from datasets import load_from_disk, logging
from torch import device, nn, no_grad, optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import CIFAR10, CIFAR100
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

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
        device: device,
    ) -> nn.Module: ...

    def build_optimizer(
        self,
        model: nn.Module,
        hyperparameter: H_contra,
        checkpoint: Checkpoint,
        device: device,
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
        device: device,
    ) -> None: ...

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: device,
    ) -> float: ...


class BertSST2Task(TaskStrategy[BertHyperparameter]):
    def build_model(
        self,
        hyperparameter: BertHyperparameter,
        checkpoint: Checkpoint,
        device: device,
    ) -> nn.Module:
        model_name: str = "google/mobilebert-uncased"
        num_labels: int = 2

        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
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
        device: device,
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
        hyperparameter = cast("BertHyperparameter", hyperparameter)

        dataset = load_from_disk(
            str(Path("~/Documents/hf_cache/sst2_arrow").expanduser()),
        )

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        def tokenize_fn(batch: dict) -> dict[str, Any]:
            return tokenizer(
                batch["sentence"],
                padding="max_length",
                truncation=True,
                max_length=hyperparameter.max_seq_length,
            )

        dataset = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=["sentence", "idx"],
        )

        dataset = dataset.rename_column("label", "labels")
        dataset.set_format(  # type: ignore[]
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )

        train_loader = DataLoader(
            dataset["train"],  # type:ignore[]
            batch_size=hyperparameter.batch_size,
            shuffle=True,
            pin_memory=False,
        )

        valid_loader = DataLoader(
            dataset["validation"],  # type:ignore[]
            batch_size=hyperparameter.batch_size,
            shuffle=False,
            pin_memory=False,
        )

        test_loader = DataLoader(
            dataset["test"],  # type:ignore[]
            batch_size=hyperparameter.batch_size,
            shuffle=False,
            pin_memory=False,
        )

        return train_loader, valid_loader, test_loader

    def train_step(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        dataloader: DataLoader,
        device: device,
    ) -> None:
        model.train()
        for raw_batch in islice(dataloader, 8):
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
        device: device,
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
        device: device,
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
        device: device,
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
        device: device,
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
        device: device,
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
        device: device,
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
        device: device,
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
        device: device,
    ) -> None:
        model.train()
        criterion = nn.CrossEntropyLoss().to(device)

        for raw_inputs, raw_targets in dataloader:
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
        device: device,
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
