"""
Manage batch bool
"""

import time

import azure.batch.models as batchmodels
from azure.batch import BatchServiceClient


def resize_pool(
    batch_service_client: BatchServiceClient,
    pool_id: str,
    dedicated_nodes: int,
    low_priority_nodes: int,
) -> None:
    """
    Resize's the number of nodes in a pool

      * batch_service_client: A Batch service client.
      * pool_id: The ID for the pool.
      * dedicated_nodes/low_priority_nodes: how many nodes of each type should this pool have
    """

    batch_service_client.pool.resize(
        pool_id=pool_id,
        pool_resize_parameter=batchmodels.PoolResizeParameter(
            target_dedicated_nodes=dedicated_nodes,
            target_low_priority_nodes=low_priority_nodes,
        ),
    )
    wait_for_pool_resize(batch_service_client, pool_id)


def wait_for_pool_resize(
    batch_service_client: BatchServiceClient, pool_id: str
) -> None:
    """
    Waits for a pool to finish resizing and all nodes to be ready

      * batch_service_client: A Batch service client.
      * pool_id: The ID for the pool.
    """

    print("waiting for nodes to be allocated")
    pool = lambda: batch_service_client.pool.get(pool_id)
    while pool().allocation_state != "steady":
        time.sleep(1)

    nodes = lambda: [
        n
        for n in batch_service_client.compute_node.list(pool_id)
        if n.state == "starting"
    ]
    print("nodes allocated, waiting for vm's to be ready")
    while not nodes():
        time.sleep(1)
    print("pool ready")
