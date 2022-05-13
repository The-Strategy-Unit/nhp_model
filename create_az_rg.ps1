# make sure to install the Azure Powershell module
# (https://docs.microsoft.com/en-us/powershell/azure/install-az-ps?view=azps-7.5.0)
# > Install-Module -Name Az -Scope CurrentUser -Repository PSGallery -Force

# you need to connect to Azure after installing the module
# > Connect-AzAccount

# you will also need to install the Azure CLI
# (https://docs.microsoft.com/en-us/cli/azure/install-azure-cli-windows?tabs=azure-cli)
# once installed you will need to login to Azure
# > az login
Function New-NhpResourceGroup {
  Param(
    [string]$subscription,
    [string]$region,
    [string]$name,
    [string]$environment
  )

  ################################################################################
  # create variables for the names of the resources we create
  ################################################################################
  $rgname = $name

  $vnetname = "$rgname-vnet"

  $kvname = "$($rgname)-kyv"

  $saname = "$($rgname)sa"
  $sakeyname = "$($saname)-key"

  $batchname = "$($rgname)batch"

  $cosmosname = "$($rgname)-cosmosdb-sql"
  $cosmoskeyname = "$($cosmosname)-key"

  $synapsename = "$($rgname)-synapse"
  $synapsekeyname = "$($synapsename)-key"

  $aaname = "$($rgname)-aa"

  $mypublicip = (Invoke-WebRequest -Uri 'http://ifconfig.me/ip').Content

  $tags = @{environment = $environment }

  ################################################################################
  # connect to azure + subscription
  ################################################################################
  Set-AzContext -Subscription $subscription

  ################################################################################
  # create the resource group
  ################################################################################
  New-AzResourceGroup `
    -Location $region `
    -Name $rgname `
    -Tag $tags

  ################################################################################
  # create the vnet
  ################################################################################
  $vnet = New-AzVirtualNetwork `
    -Location $region `
    -ResourceGroupName $rgname `
    -Name $vnetname `
    -AddressPrefix '10.1.0.0/16' `
    -Tag $tags

  Add-AzVirtualNetworkSubnetConfig `
    -VirtualNetwork $vnet `
    -Name 'default' `
    -AddressPrefix '10.1.0.0/22'

  Add-AzVirtualNetworkSubnetConfig `
    -VirtualNetwork $vnet `
    -Name 'batch' `
    -AddressPrefix '10.1.4.0/22' `
    -ServiceEndpoint @('Microsoft.AzureCosmosDB', 'Microsoft.Storage')

  $vnet = Set-AzVirtualNetwork -VirtualNetwork $vnet

  ################################################################################
  # create keyvault
  ################################################################################
  $kvnrs = New-AzKeyVaultNetworkRuleSetObject `
    -DefaultAction Deny `
    -IpAddressRange $mypublicip

  $kv = New-AzKeyVault `
    -Location $region `
    -ResourceGroupName $rgname `
    -Name $kvname `
    -NetworkRuleSet $kvnrs `
    -Sku 'Standard' `
    -EnablePurgeProtection `
    -Tag $tags

  # create keys
  Add-AzKeyVaultKey `
    -VaultName $kvname `
    -Name $sakeyname `
    -Destination 'Software' `
    -Size 4096 `
    -KeyType RSA

  Add-AzKeyVaultKey `
    -VaultName $kvname `
    -Name $synapsekeyname `
    -Destination 'Software' `
    -Size 3072 `
    -KeyType RSA

  Add-AzKeyVaultKey `
    -VaultName $kvname `
    -Name $cosmoskeyname `
    -Destination 'Software' `
    -Size 4096 `
    -KeyType RSA

  ################################################################################
  # create the storage account
  ################################################################################
  $sa = New-AzStorageAccount `
    -Location $region `
    -ResourceGroupName $rgname `
    -Name $saname `
    -Kind StorageV2 `
    -SkuName Standard_LRS `
    -AccessTier Hot `
    -AllowBlobPublicAccess $false `
    -EnableHierarchicalNamespace $true `
    -AssignIdentity `
    -NetworkRuleSet (@{
      bypass              = 'AzureServices';
      virtualNetworkRules = (@{
          VirtualNetworkResourceId = $batchsubnet.id;
          Action                   = 'allow'
        });
      ipRules             = (@{
          IPAddressOrRange = $mypublicip
        });
      defaultAction       = 'Deny'
    }) `
    -Tag $tags

  # grant Key Vault access to Azure Storage
  Set-AzKeyVaultAccessPolicy `
    -VaultName $kvname `
    -ObjectId $sa.Identity.PrincipalId `
    -PermissionsToKeys get, unwrapKey, wrapKey

  # encrypt the storage account
  $sa = Set-AzStorageAccount `
    -ResourceGroupName $rgname `
    -AccountName $saname `
    -KeyvaultEncryption `
    -KeyName $sakeyname `
    -KeyVaultUri $kv.VaultUri

  # create containers
  'app', 'data', 'results', 'queue' | ForEach-Object -Process {
    New-AzStorageContainer -Name $_ -Context $sa.Context -Permission Off
  }

  # create fileshare
  New-AzRmStorageShare `
    -ResourceGroupName $rgname `
    -StorageAccountName $saname `
    -Name 'batch' `
    -AccessTier TransactionOptimized `
    -QuotaGiB 100

  ################################################################################
  # create the batch account
  ################################################################################
  New-AzBatchAccount `
    -Location $region `
    -ResourceGroupName $rgname `
    -Name $batchname `
    -PublicNetworkAccess Enabled `
    -AutoStorageAccountId $sa.id `
    -Tag $tags

  # create our pool
  Add-NhpBatchPool $rgname $vnetname $saname $batchname

  ################################################################################
  # create the cosmosdb account
  ################################################################################

  # first you have to give the standard cosmos db service principal acccess to keyvault
  Set-AzKeyVaultAccessPolicy `
    -VaultName $kvname `
    -ServicePrincipalName 'a232010e-820c-4083-83bb-3ace5fc29d0b' `
    -PermissionsToKeys get, unwrapKey, wrapKey

  $CosmosDBProperties = @{
    'databaseAccountOfferType'      = 'Standard';
    'locations'                     = @(
      @{ 'locationName' = $region; 'failoverPriority' = 0 }
    );
    'keyVaultKeyUri'                = "https://$($kvname).vault.azure.net/keys/$($cosmoskeyname)";
    'capabilities'                  = @(
      @{ 'name' = 'EnableServerless' }
    );
    'backupPolicy'                  = @{
      'type'                   = 'Periodic';
      'periodicModeProperties' = @{
        'backupIntervalInMinutes'        = 240;
        'backupRetentionIntervalInHours' = 8;
        'backupStorageRedundancy'        = 'Local'
      }
    };
    'isVirtualNetworkFilterEnabled' = $true;
    'virtualNetworkRules'           = @(
      @{ 'id' = $batchsubnet.id }
    )
  }

  New-AzResource `
    -ResourceType 'Microsoft.DocumentDb/databaseAccounts' `
    -ResourceGroupName $rgname `
    -Location $region `
    -Name $cosmosname `
    -PropertyObject $CosmosDBProperties `
    -Force `
    -Tag $tags

  # set default identity to System Assigned Identity
  # cannot see how to do this with powershell, so using az cli
  $cosmosid = (az cosmosdb update `
      --resource-group $rgname `
      --name $cosmosname `
      --default-identity 'SystemAssignedIdentity' |
    ConvertFrom-Json |
    Select-Object -ExpandProperty Identity |
    Select-Object -ExpandProperty principalId
  )

  # grant access to the System Assigned Identity
  Set-AzKeyVaultAccessPolicy `
    -VaultName $kvname `
    -ObjectId $cosmosid `
    -PermissionsToKeys get, unwrapKey, wrapKey

  # remove the standard azure cosmos db service principal
  Remove-AzKeyVaultAccessPolicy `
    -VaultName $kvname `
    -ServicePrincipalName 'a232010e-820c-4083-83bb-3ace5fc29d0b'

  # enable Synapse Link
  Update-AzCosmosDBAccount `
    -ResourceGroupName $rgname `
    -Name $cosmosname `
    -EnableAnalyticalStorage true

  ################################################################################
  # create synapse workspace
  ################################################################################
  # create a password to use for synapse analytics and store it in keyvault
  # e.g. use https://www.grc.com/passwords.htm to generate a very strong 63 character long password
  $password = Read-Host 'Enter a password for Synapse Analytics Admin Account' -AsSecureString
  Set-AzKeyVaultSecret -VaultName $kvname -Name "$($synapsename)-admin-password" -SecretValue $secretvalue
  $synapsecred = New-Object System.Management.Automation.PSCredential ("$($rgname)_admin", $password)

  $synapse = New-AzSynapseWorkspace `
    -Location $region `
    -ResourceGroupName $rgname `
    -Name $synapsename `
    -DefaultDataLakeStorageAccountName $saname `
    -DefaultDataLakeStorageFilesystem 'data' `
    -ManagedResourceGroupName "$($rgname)-synapse-mrg" `
    -SqlAdministratorLoginCredential $synapsecred `
    -EncryptionKeyIdentifier "https://$($kvname).vault.azure.net/keys/$($synapsekeyname)" `
    -Tag $tags
    
  Set-AzKeyVaultAccessPolicy `
    -VaultName $kvname `
    -ObjectId $synapse.Identity.PrincipalId `
    -PermissionsToKeys get, unwrapKey, wrapKey

  Enable-AzSynapseWorkspace `
    -ResourceGroupName $rgname `
    -WorkspaceName $synapsename

  ################################################################################
  # Create an automation account
  ################################################################################
  New-AzAutomationAccount `
    -ResourceGroupName $rgname `
    -Name $aaname `
    -Location $region `
    -AssignSystemIdentity `
    -DisablePublicNetworkAccess `
    -Plan 'Free' `
    -Tags $tags
}

Function Add-NhpBatchPool {
  Param(
    [string]$rgname,
    [string]$vnetname = "$($rgname)-vnet",
    [string]$saname = "$($rgname)sa",
    [string]$batchname = "$($rgname)batch",
    [string]$vmsize = 'STANDARD_D16D_V5'
  )

  $poolname = "$($rgname)-model"
  # get the batch account
  $batch = Get-AzBatchAccount -AccountName $batchname

  # get the batch subnet
  $vnet = Get-AzVirtualNetwork -ResourceGroupName $rgname -Name $vnetname
  $batchsubnet = Get-AzVirtualNetworkSubnetConfig -Name 'batch' -VirtualNetwork $vnet

  # pool vnet config
  $poolvnetConfig = New-Object Microsoft.Azure.Commands.Batch.Models.PSNetworkConfiguration
  $pip = New-Object Microsoft.Azure.Commands.Batch.Models.PSPublicIPAddressConfiguration -ArgumentList @('BatchManaged')
  $poolvnetConfig.publicIPAddressConfiguration = $pip
  $poolvnetConfig.SubnetId = $batchsubnet.id

  # create the virtual machine image reference
  $imageReference = New-Object `
    -TypeName 'Microsoft.Azure.Commands.Batch.Models.PSImageReference' `
    -ArgumentList @('0001-com-ubuntu-server-focal', 'canonical', '20_04-lts-gen2', 'latest')
  $configuration = New-Object `
    -TypeName 'Microsoft.Azure.Commands.Batch.Models.PSVirtualMachineConfiguration' `
    -ArgumentList @($imageReference, 'batch.node.ubuntu 20.04')

  # mount the storage account blob containers
  $sakey = [Microsoft.Azure.Batch.AzureStorageAuthenticationKey]::FromAccountKey(
    (Get-AzStorageAccountKey -ResourceGroupName $rgname -Name $saname).Value[0]
  )
  $blobs = 'app', 'data', 'results', 'queue' | ForEach-Object -Process {
    New-Object `
      -TypeName 'Microsoft.Azure.Commands.Batch.Models.PSAzureBlobFileSystemConfiguration' `
      -ArgumentList @($saname, $_, $_, $sakey, ''
    )
  }
  $fileshares = @('batch') | ForEach-Object -Process {
    $fso = '-o vers=3.0,dir_mode=0777,file_mode=0777,sec=ntlmssp'
    New-Object `
      -TypeName 'Microsoft.Azure.Commands.Batch.Models.PSAzureFileShareConfiguration' `
      -ArgumentList @($saname, "https://$($saname).file.core.windows.net/$($_)", $_, $sakey.AccountKey, $fso
    )
  }

  $mountConfig = ($blobs + $fileshares) | ForEach-Object {
    New-Object -TypeName 'Microsoft.Azure.Commands.Batch.Models.PSMountConfiguration' -ArgumentList $_
  }

  # define the pool start task
  $startTasks = (
    'mkdir /opt/nhp',
    'tar -xzf /mnt/batch/tasks/fsmounts/app/nhp.tar.gz -C /opt/nhp'
  ) -Join '; '

  $startTaskReference = New-Object Microsoft.Azure.Commands.Batch.Models.PSStartTask
  $startTaskReference.CommandLine = "/bin/bash -c 'set -eo pipefail; $($startTasks)'"
  $startTaskReference.UserIdentity = New-Object Microsoft.Azure.Commands.Batch.Models.PSAutoUserSpecification -ArgumentList @('Pool', 'Admin')
  $startTaskReference.WaitForSuccess = $true
  $startTaskReference.MaxTaskRetryCount = 0

  # if the pool already exists, delete it
  if ((Get-AzBatchPool -BatchContext $batch -Filter "id eq '$($poolname)'").Count -eq 1) {
    Remove-AzBatchPool -Id $poolname -BatchContext $batch -Force 2>$null
    # wait 30 seconds for the pool to be deleted
    Write-Host 'Deleted old pool, waiting 30s to create new pool...'
    Start-Sleep -s 30
  }

  # create the pool
  New-AzBatchPool `
    -Id $poolname `
    -VirtualMachineSize $vmsize `
    -VirtualMachineConfiguration $configuration `
    -TargetDedicatedComputeNodes 0 `
    -NetworkConfiguration $poolvnetConfig `
    -MountConfiguration @($mountConfig) `
    -StartTask $startTaskReference `
    -BatchContext $batch
}
