import time
import azure.batch.models as batchmodels
import config

def create_pool(batch_service_client, pool_id):
  vm = batchmodels.VirtualMachineConfiguration(
    image_reference = batchmodels.ImageReference(
      publisher = "microsoft-azure-batch",
      offer = "ubuntu-server-container",
      sku = "20-04-lts",
      version = "latest"
    ),
    node_agent_sku_id = "batch.node.ubuntu 20.04",
    container_configuration = batchmodels.ContainerConfiguration(
      container_image_names = [config._CR_CONTAINER_NAME],
      container_registries = [
        batchmodels.ContainerRegistry(
          user_name = config._CR_REGISTRY_NAME,
          password = config._CR_PASSWORD,
          registry_server = config._CR_LOGIN_SERVER
        )
      ]
    )
  )

  mount = lambda container: batchmodels.MountConfiguration(
    azure_blob_file_system_configuration = batchmodels.AzureBlobFileSystemConfiguration(
      account_name = config._STORAGE_ACCOUNT_NAME,
      account_key = config._STORAGE_ACCOUNT_KEY,
      container_name = container,
      relative_mount_path = container
    )
  )

  pool = batchmodels.PoolAddParameter(
    id = pool_id,
    vm_size = config._POOL_VM_SIZE,
    virtual_machine_configuration = vm,
    mount_configuration =  [mount(x) for x in ["data", "queue", "results"]],
    target_dedicated_nodes = config._POOL_NODE_DEDICATED_COUNT,
    target_low_priority_nodes = config._POOL_NODE_LOW_PRIORITY_COUNT
  )
  batch_service_client.pool.add(pool)
  wait_for_pool_resize(batch_service_client, pool_id)

def resize_pool(batch_service_client, pool_id, dedicated_nodes, low_priority_nodes):
  batch_service_client.pool.resize(
    pool_id = pool_id,
    pool_resize_parameter = batchmodels.PoolResizeParameter(
      target_dedicated_nodes = dedicated_nodes,
      target_low_priority_nodes = low_priority_nodes
    )
  )
  wait_for_pool_resize(batch_service_client, pool_id)

def delete_pool(batch_service_client, pool_id):
  batch_service_client.pool.delete(pool_id)
  print ("pool deleting: waiting for nodes to be deallocated")
  while batch_service_client.pool.exists(pool_id):
    time.sleep(1)
  print ("pool deleted")

def wait_for_pool_resize(batch_service_client, pool_id):
  print ("waiting for nodes to be allocated")
  ps = lambda: batch_service_client.pool.get(pool_id)
  while ps().allocation_state != "steady":
    time.sleep(1)

  ns = lambda: [n for n in batch_service_client.compute_node.list(pool_id) if n.state == "starting"]
  print ("nodes allocated, waiting for vm's to be ready")
  while not ns():
    time.sleep(1)
  print ("pool ready")