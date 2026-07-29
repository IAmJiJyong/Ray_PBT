import unittest

from src.trial_state import Hyperparameter, TrialState
from src.utils import ModelType, Checkpoint


class TestTrialState(unittest.TestCase):
    def test_trial_state_without_checkpoint(self) -> None:
        trial = TrialState(
            id=0,
            hyperparameter=Hyperparameter(0.1, 128),
            checkpoint=Checkpoint.empty(),
            generation=0,
        )

        assert trial.checkpoint.is_empty(), "Trial checkpoint is not None"
        assert trial.snapshot.checkpoint.is_empty(), "Trial is not without checkpoint"


if __name__ == "__main__":
    unittest.main()
