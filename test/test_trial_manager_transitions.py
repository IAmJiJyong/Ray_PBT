import unittest
import ray
from src.trial_manager import TrialManager
from src.trial_state import TrialState, Hyperparameter
from src.utils import TrialStatus, ModelType, Checkpoint


class TestTrialManagerTransitions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray.init(ignore_reinit_error=True, num_cpus=1)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def setUp(self):
        # Mock TensorBoard Manager
        @ray.remote
        class MockTB:
            def add_best_acc_to_global(self, *args, **kwargs):
                pass

        self.tb = MockTB.remote()

        self.trial1 = TrialState(
            id=1,
            hyperparameter=Hyperparameter(0.1, 128),
            checkpoint=Checkpoint.empty(),
            generation=0,
        )
        self.tm = TrialManager.remote([self.trial1], self.tb)

    def test_transition_running_to_waiting(self):
        # Initial: PENDING
        # PENDING -> WAITING (acquire_pending_trials)
        ray.get(self.tm.transition_status.remote(1, TrialStatus.WAITING))

        # WAITING -> RUNNING
        ray.get(self.tm.transition_status.remote(1, TrialStatus.RUNNING))

        running_ids = ray.get(self.tm.get_running_ids.remote())
        waiting_ids = ray.get(self.tm.get_waiting_ids.remote())
        assert 1 in running_ids
        assert 1 not in waiting_ids

        # RUNNING -> WAITING (e.g. paused or preempted)
        ray.get(self.tm.transition_status.remote(1, TrialStatus.WAITING))

        running_ids = ray.get(self.tm.get_running_ids.remote())
        waiting_ids = ray.get(self.tm.get_waiting_ids.remote())

        # This is where we expect the bug: 1 might still be in running_ids
        print(f"Running IDs: {running_ids}")
        print(f"Waiting IDs: {waiting_ids}")

        self.assertIn(1, waiting_ids)
        self.assertNotIn(
            1,
            running_ids,
            "Trial 1 should not be in running_ids after transition to WAITING",
        )

    def test_transition_waiting_to_pending(self):
        # Initial: PENDING
        # PENDING -> WAITING
        ray.get(self.tm.transition_status.remote(1, TrialStatus.WAITING))

        # WAITING -> PENDING
        ray.get(self.tm.transition_status.remote(1, TrialStatus.PENDING))

        waiting_ids = ray.get(self.tm.get_waiting_ids.remote())
        pending_ids = ray.get(self.tm.get_pending_ids.remote())

        self.assertIn(1, pending_ids)
        self.assertNotIn(
            1,
            waiting_ids,
            "Trial 1 should not be in waiting_ids after transition to PENDING",
        )

    def test_transition_to_failed(self):
        # PENDING -> FAILED
        ray.get(self.tm.transition_status.remote(1, TrialStatus.FAILED))

        failed_ids = ray.get(self.tm.get_failed_ids.remote())
        pending_ids = ray.get(self.tm.get_pending_ids.remote())

        self.assertIn(1, failed_ids)
        self.assertNotIn(1, pending_ids)

        # FAILED -> PENDING (retry)
        ray.get(self.tm.transition_status.remote(1, TrialStatus.PENDING))

        failed_ids = ray.get(self.tm.get_failed_ids.remote())
        pending_ids = ray.get(self.tm.get_pending_ids.remote())

        self.assertNotIn(1, failed_ids)
        self.assertIn(1, pending_ids)


if __name__ == "__main__":
    unittest.main()
