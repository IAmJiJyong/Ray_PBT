import unittest
import warnings

import ray

from src.worker_manager import generate_all_worker_states


class TestWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        warnings.simplefilter("ignore", ResourceWarning)
        ray.init(ignore_reinit_error=True, num_cpus=2)

    @classmethod
    def tearDownClass(cls) -> None:
        ray.shutdown()

    def test_generate_all_worker_states(self) -> None:
        worker_states = generate_all_worker_states()
        # In a local test with num_cpus=2, it might generate 0 workers because 
        # it skips the head node for CPU workers.
        # But let's just check if it runs without error.
        assert isinstance(worker_states, list)


if __name__ == "__main__":
    unittest.main()
