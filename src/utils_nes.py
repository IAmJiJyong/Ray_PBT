import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import repeat
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy import exp, floating
from numpy.linalg import cholesky, det, inv
from numpy.random import multivariate_normal
from numpy.typing import NDArray
from scipy.linalg import expm

from .utils import Hyperparameter, ModelType

# ╭──────────────────────────────────────────────────────────╮
# │                           Enum                           │
# ╰──────────────────────────────────────────────────────────╯

class TuneType(Enum):
    PBT = auto()
    NES = auto()

    def __str__(self) -> str:
        return self.name

# ╭──────────────────────────────────────────────────────────╮
# │                        DataClass                         │
# ╰──────────────────────────────────────────────────────────╯


@dataclass(slots=True)
class Fitness:
    accuracy: float
    hyperparameter: Hyperparameter


@dataclass(slots=True)
class NaturalGradients(NamedTuple):
    nabla_sigma: floating
    nabla_delta: NDArray[floating]
    nabla_b: NDArray[floating]

    def __str__(self) -> str:
        return (
            f"NaturalGradients(nabla_sigma: {self.nabla_sigma:.3f}, "
            f"nabla_delta: {self.nabla_delta}, "
            f"nabla_b: {self.nabla_b})"
        )


@dataclass(slots=True)
class Distribution:
    mean: NDArray[floating]
    sigma: float
    b_matrix: NDArray[floating]
    fitnesses: deque[Fitness]
    previous_gradients: NaturalGradients | None
    log_file: Path = field(init=False)

    @staticmethod
    def compute_kl_divergence(
        p: "Distribution",
        q: "Distribution",
    ) -> float:
        r"""
        Compute the KL divergence between two distributions.

        $$
        \operatorname{KL}(p | q) = \frac{1}{2}\left(\log{\frac{\Sigma_q}
        {\Sigma_p}} - d + \operatorname{tr}
        \left(\Sigma_q^{-1}\Sigma_p\right) + \left(\mu_q - \mu_p\right)^T
        \Sigma_q^{-1} \left(\mu_q - \mu_p\right)\right)
        $$

        Args:
            p (Distribution): The first distribution.
            q (Distribution): The second distribution.

        Returns:
            flost: The KL divergence between the two distributions.
        """
        mean_p, covariance_p = p.get_attribute()
        mean_q, covariance_q = q.get_attribute()

        log_term = np.log(det(covariance_q) / det(covariance_p))
        d = mean_p.shape[0]
        trace_term = np.trace(inv(covariance_q) @ covariance_p)
        diffence_term = mean_q - mean_p

        res = (
            log_term
            - d
            + trace_term
            + diffence_term.T @ inv(covariance_q) @ diffence_term
        )

        return res / 2

    @staticmethod
    def fitness_shaping(fitnesses: list[Fitness]) -> list:
        """
        Fitness Shaping use

        Args:
            fitness: model's fitness.

        Returns:
            List[float]: Returns utility.

        Raises:
            ZeroDivisionError: if length of fitnesses is zero.
        """
        #                            ╭───────────────────────╮
        #                            │ NES'S Fitness Shaping │
        #                            ╰───────────────────────╯
        size = len(fitnesses)
        if size == 0:
            raise ZeroDivisionError
        base = math.log(size / 2 + 1)
        utilities_raw = [max(0.0, base - math.log(i + 1)) for i in range(size)]
        denominator = sum(utilities_raw)
        return [u / denominator - 1.0 / size for u in utilities_raw]

    @property
    def covariance(self) -> NDArray[floating]:
        return (self.b_matrix.T @ self.b_matrix) * (self.sigma**2)

    def get_attribute(
        self,
    ) -> tuple[NDArray[floating], NDArray[floating]]:
        return (self.mean, self.covariance)

    def __post_init__(self) -> None:
        log_dir_path = "~/Documents/workspace/shaogu/Ray_PBT/log/"
        log_dir = Path(log_dir_path).expanduser()
        log_dir.mkdir(exist_ok=True)
        self.log_file = log_dir / "nes.log"

    @classmethod
    def get_random_distribution(cls) -> "Distribution":
        init_hyper = np.concatenate(
            [
                np.random.uniform(  # noqa:NPY002
                    1e-4,
                    1e-1,
                    size=2,
                ),
                np.random.uniform(  # noqa:NPY002
                    1e-7,
                    1e-6,
                    size=2,
                ),
                np.random.uniform(  # noqa:NPY002
                    0.0,
                    1.0,
                    size=1,
                ),
            ],
        )

        diag_vals = (
            np.random.uniform(  # noqa: NPY002
                low=0.1,
                high=0.333,
                size=len(init_hyper),
            )
            * init_hyper
        )
        covariance = np.diag(diag_vals)
        square_root_of_covariance = cholesky(covariance).T
        mean = np.array(init_hyper)
        sigma = abs(det(square_root_of_covariance)) ** (1 / len(init_hyper))
        b_matrix = square_root_of_covariance / sigma
        population_size = 4 + int(3 * np.log(len(mean)))

        return cls(
            mean=mean,
            sigma=sigma,
            b_matrix=b_matrix,
            fitnesses=deque(maxlen=population_size),
            previous_gradients=None,
        )

    def get_new_hyper(self) -> Hyperparameter:
        modified_sample: np.ndarray | None = None
        while modified_sample is None or np.any(
            (modified_sample <= 0.0) | (modified_sample >= 1.0),
        ):
            sample: NDArray[floating] = multivariate_normal(  # noqa: NPY002
                mean=np.zeros(self.mean.size),
                cov=np.eye(self.mean.size),
            )
            modified_sample = self.mean + self.sigma * (self.b_matrix @ sample)
        parameter: list[float] = modified_sample.tolist()
        return Hyperparameter(
            *parameter[:-1],
            batch_size=int(parameter[-1] * 480.0 + 32),
            model_type=ModelType.RESNET_18,
        )

    def update_distribution(self, fitness: Fitness) -> None:
        self.fitnesses.append(fitness)
        dimension = len(self.mean)
        population_size = 4 + int(3 * np.log(dimension))
        if len(self.fitnesses) < population_size:
            if len(self.fitnesses) == 1:
                with self.log_file.open("w") as f:
                    message = (
                        "\033[1;91mTune Start Distribution\033[0m\n"
                        f"Distribution(mean:{self.mean}"
                        f", sigma:{self.sigma}, b_matrix:{self.b_matrix})"
                    )
                    f.write(message)
                    f.write("\n")
            return

        eta_mean = 1.0 / dimension
        eta_b_matrix = (
            (9 + 3 * np.log(dimension))
            / (5 * dimension * np.sqrt(dimension))
            / dimension
        )
        eta_sigma = (
            (9 + 3 * np.log(dimension))
            / (5 * dimension * np.sqrt(dimension))
            / dimension
        )

        gradients: NaturalGradients = self._compute_gradient()
        self.mean = self.mean + (
            eta_mean * self.sigma * (self.b_matrix @ gradients.nabla_delta)
        )
        self.sigma = self.sigma * exp((eta_sigma / 2) * gradients.nabla_sigma)
        self.b_matrix = self.b_matrix @ expm(
            (eta_b_matrix / 2) * gradients.nabla_b,
        )
        with self.log_file.open("a") as f:
            message = (
                f"Distribution(mean:{self.mean}"
                f", sigma:{self.sigma}, b_matrix:{self.b_matrix})"
            )

            f.write(message)
            f.write("\n")

    def _compute_gradient(
        self,
    ) -> NaturalGradients:
        """
        Compute the natural gradient from the fitnesses.

        Returns:
            NaturalGradients: The computed natural gradients.
        """
        sorted_fitnesses = sorted(
            self.fitnesses,
            key=lambda x: x.accuracy,
            reverse=True,
        )
        utilities = Distribution.fitness_shaping(sorted_fitnesses)
        shaping_fitnesses = [
            (utility, fitness.hyperparameter)
            for utility, fitness in zip(
                utilities,
                sorted_fitnesses,
                strict=True,
            )
        ]
        delta_gradient = np.sum(
            [u * h.to_ndarray() for u, h in shaping_fitnesses],
            axis=0,
        )
        m_gradient = np.sum(
            [
                u
                * (
                    np.outer(
                        h.to_ndarray(),
                        h.to_ndarray(),
                    )
                    - np.eye(self.mean.size)
                )
                for u, h in shaping_fitnesses
            ],
            axis=0,
        )
        sigma_gradient = np.trace(m_gradient) / self.mean.size
        b_gradient = m_gradient - sigma_gradient * np.eye(self.mean.size)

        return NaturalGradients(
            nabla_sigma=sigma_gradient,
            nabla_delta=delta_gradient,
            nabla_b=b_gradient,
        )


# ╭──────────────────────────────────────────────────────────╮
# │                        NamedTuple                        │
# ╰──────────────────────────────────────────────────────────╯


class Group(NamedTuple):
    distribution: Distribution
    trials: set[int]


class DistributionManager:
    __slots__ = (
        "counter",
        "distributions",
    )

    def __init__(self, num_distributions: int) -> None:
        self.distributions: list[Group] = []
        self.counter = 0

        for _ in repeat(None, num_distributions):
            self.distributions.append(
                Group(
                    distribution=Distribution.get_random_distribution(),
                    trials=set(),
                ),
            )

    def get_new_hyper(self, trial_id: int) -> Hyperparameter:
        for distribution, trials in self.distributions:
            if trial_id in trials:
                return distribution.get_new_hyper()

        self.distributions.sort(key=lambda x: len(x.trials))
        self.distributions[0].trials.add(trial_id)
        return self.distributions[0].distribution.get_new_hyper()

    def update_distribution(self, trial_id: int, fitness: Fitness) -> None:
        for distribution, trials in self.distributions:
            if trial_id in trials:
                distribution.update_distribution(fitness)
                return

        error_msg = f"Trial ID {trial_id} not found in any distribution."
        raise ValueError(error_msg)

    def hyper_init(self, trial_id: int) -> Hyperparameter:
        size = len(self.distributions)
        distribution = self.distributions[self.counter % size]
        distribution.trials.add(trial_id)
        return distribution.distribution.get_new_hyper()
