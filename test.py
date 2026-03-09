from pathlib import Path

import ray
from datasets import load_dataset
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

ray.init(address="auto")

CACHE_ROOT = "~/Documents/hf_cache"


@ray.remote
def download_on_node() -> str:
    cache_dir = Path(CACHE_ROOT).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "glue",
        "sst2",
        cache_dir=str(cache_dir),
    )

    ds.save_to_disk(str(cache_dir / "sst2_arrow"))

    return f"done on {ray.get_runtime_context().node_id.hex()}"


nodes = [n["NodeID"] for n in ray.nodes() if n["Alive"]]

tasks = []

for node_id in nodes:
    strat = NodeAffinitySchedulingStrategy(
        node_id=node_id,
        soft=False,
    )

    tasks.append(download_on_node.options(scheduling_strategy=strat).remote())

ray.get(tasks)

print("✅ All nodes finished downloading SST-2")
