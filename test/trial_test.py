import os
import sys
from typing import List
from itertools import islice

import ray
import torch.nn as nn
import torch
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from main_bak import Trial
from trial import Hyperparameter, TrialScheduler, TrialState
from utils import ModelType
from worker import generate_all_workers


def train_step(model, optimizer, train_loader, batch_size, device=torch.device("cpu")):
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)

    for inputs, targets in islice(train_loader, 1024 // batch_size):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


@ray.remote
def test():
    return os.listdir(os.getcwd())


def generate_trial_states(n: int = 10) -> List[TrialState]:
    return [
        TrialState(
            i,
            Hyperparameter(
                lr=random.uniform(0.001, 1),
                momentum=random.uniform(0.001, 1),
                batch_size=random.choice([64, 128, 256, 512, 1024]),
                model_type=ModelType.RESNET_18,
            ),
            stop_iteration=100,
        )
        for i in range(n)
    ]


if __name__ == "__main__":
    ray.init(runtime_env={"working_dir": "./src"})

    print(ray.get(test.remote()))

    workers = generate_all_workers(train_step)
    print(*workers, sep="\n")
    worker = workers[0]

    trial_states = generate_trial_states()
    print(f"總共{len(trial_states)} 個 Trial")
    print(*[t.hyperparameter for t in trial_states], sep="\n")

    scheduler = TrialScheduler(trial_states, workers)
    scheduler.run()
