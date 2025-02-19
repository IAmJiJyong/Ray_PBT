from dataclasses import dataclass
from typing import List

import ray


@dataclass
class Worker:
    id: int
    num_cpus: int
    num_gpus: int
    node_name: str
    calculate_ability: float
    used_count: float


def generate_all_workers() -> List[Worker]:
    visited_address = set()
    workers = []
    index = 0

    for node in ray.nodes():
        if node["Alive"]:
            if node["NodeManagerAddress"] in visited_address:
                continue

            resource = node["Resources"]
            if "CPU" in resource:
                workers.append(
                    Worker(
                        id=index,
                        num_cpus=resource.get("CPU", 0),
                        num_gpus=0,
                        node_name=f"node:{node['NodeManagerAddress']}",
                        calculate_ability=0,
                        used_count=0,
                    )
                )
                index += 1
            if "GPU" in resource:
                workers.append(
                    Worker(
                        id=index,
                        num_cpus=0,
                        num_gpus=resource.get("GPU", 0),
                        node_name=f"node:{node['NodeManagerAddress']}",
                        calculate_ability=0,
                        used_count=0,
                    )
                )
                index += 1
            visited_address.add(node["NodeManagerAddress"])

    return workers
