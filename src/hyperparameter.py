import random
from dataclasses import dataclass

from .utils import Hyperparameter


@dataclass
class CNNHyperparameter(Hyperparameter):
    lr: float
    momentum: float
    batch_size: int
    cpu_batch_size: int

    def __str__(self) -> str:
        return (
            f"Hyperparameter(lr:{self.lr:.3f}, momentum:{self.momentum:.3f}, "
            f"batch_size:{self.batch_size:4d}"
        )

    @classmethod
    def random(cls) -> "CNNHyperparameter":
        return cls(
            lr=random.uniform(0.001, 0.1),
            momentum=random.uniform(0.001, 1),
            batch_size=random.choice([32, 64, 128]),
            cpu_batch_size=32,
        )

    def explore(self) -> "CNNHyperparameter":
        momentum = self.momentum * 1.2
        if momentum > 1.0:
            momentum = self.momentum * 0.8

        lr = self.lr * 0.8
        lr_lower_bound = 1e-7
        if lr < lr_lower_bound:
            lr = self.lr * 1.2

        return CNNHyperparameter(
            lr=lr,
            momentum=momentum,
            batch_size=self.batch_size,
            cpu_batch_size=self.cpu_batch_size,
        )


@dataclass
class BertHyperparameter(Hyperparameter):
    """
    一個用於微調 BERT 於 SST-2 分類任務的具體超參數實作。
    """

    lr: float
    batch_size: int

    weight_decay: float
    adam_epsilon: float
    warmup_steps: int
    max_seq_length: int
    cpu_batch_size: int

    def __str__(self) -> str:
        return (
            f"lr={self.lr:.1e}, "
            f"batch_size={self.batch_size}, "
            f"weight_decay={self.weight_decay:.1e}"
        )

    @classmethod
    def random(cls) -> "BertHyperparameter":
        """
        為 BERT 微調任務生成一組典型的隨機超參數。
        """
        return cls(
            lr=random.uniform(1e-5, 5e-4),
            batch_size=random.choice([64, 128, 256]),
            weight_decay=random.uniform(1e-1, 1e-4),
            adam_epsilon=1e-8,
            warmup_steps=0,
            max_seq_length=128,
            cpu_batch_size=64,
        )

    def explore(self) -> "BertHyperparameter":
        """
        對現有超參數進行微調探索。
        """
        new_lr = self.lr * random.choice([0.8, 1.2])
        new_weight_decay = self.weight_decay * random.choice([0.8, 1.2])
        new_batch_size = random.choice([64, 128, 256])

        return BertHyperparameter(
            lr=new_lr,
            batch_size=new_batch_size,
            weight_decay=new_weight_decay,
            adam_epsilon=self.adam_epsilon,
            warmup_steps=self.warmup_steps,
            max_seq_length=self.max_seq_length,
            cpu_batch_size=self.cpu_batch_size,
        )
