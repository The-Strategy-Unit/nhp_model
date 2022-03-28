import time
import batch.config as config

from azure.batch import BatchServiceClient
import azure.batch.models as batchmodels


def create_pool(batch_service_client: BatchServiceClient, pool_id: str) -> None:
    """
    Creates a pool of VM's

      * batch_service_client: A Batch service client.
      * pool_id: The ID for the pool.
    """
    vm_image = batchmodels.ImageReference(
        publisher="microsoft-azure-batch",
        offer="ubuntu-server-container",
        sku="20-04-lts",
        version="latest",
    )

    task_commands = ";".join(
        [
            "apt install -y python3-pip",
            "pip install -r /mnt/batch/tasks/fsmounts/app/requirements.txt",
        ]
    )

    start_task_conf = batchmodels.StartTask(
        command_line=f"/bin/bash -c 'set -e; set -o pipefail; {task_commands}; wait'",
        wait_for_success=True,
        user_identity=batchmodels.UserIdentity(
            auto_user=batchmodels.AutoUserSpecification(
                elevation_level=batchmodels.ElevationLevel.admin,
                scope=batchmodels.AutoUserScope.pool,
            )
        ),
    )

    vm = batchmodels.VirtualMachineConfiguration(
        image_reference=vm_image, node_agent_sku_id="batch.node.ubuntu 20.04"
    )

    mount = lambda container: batchmodels.MountConfiguration(
        azure_blob_file_system_configuration=batchmodels.AzureBlobFileSystemConfiguration(
            account_name=config._STORAGE_ACCOUNT_NAME,
            account_key=config._STORAGE_ACCOUNT_KEY,
            container_name=container,
            relative_mount_path=container,
        )
    )

    pool = batchmodels.PoolAddParameter(
        id=pool_id,
        vm_size=config._POOL_VM_SIZE,
        virtual_machine_configuration=vm,
        mount_configuration=[mount(x) for x in ["data", "queue", "app"]],
        target_dedicated_nodes=config._POOL_NODE_DEDICATED_COUNT,
        target_low_priority_nodes=config._POOL_NODE_LOW_PRIORITY_COUNT,
        start_task=start_task_conf,
    )

    batch_service_client.pool.add(pool)
    wait_for_pool_resize(batch_service_client, pool_id)


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


def delete_pool(batch_service_client: BatchServiceClient, pool_id: str) -> None:
    """
    Deletes a pool of VM's

      * batch_service_client: A Batch service client.
      * pool_id: The ID for the pool.
    """

    batch_service_client.pool.delete(pool_id)
    print("pool deleting: waiting for nodes to be deallocated")
    while batch_service_client.pool.exists(pool_id):
        time.sleep(1)
    print("pool deleted")


def wait_for_pool_resize(
    batch_service_client: BatchServiceClient, pool_id: str
) -> None:
    """
    Waits for a pool to finish resizing and all nodes to be ready

      * batch_service_client: A Batch service client.
      * pool_id: The ID for the pool.
    """

    print("waiting for nodes to be allocated")
    ps = lambda: batch_service_client.pool.get(pool_id)
    while ps().allocation_state != "steady":
        time.sleep(1)

    ns = lambda: [
        n
        for n in batch_service_client.compute_node.list(pool_id)
        if n.state == "starting"
    ]
    print("nodes allocated, waiting for vm's to be ready")
    while not ns():
        time.sleep(1)
    print("pool ready")
