$saname = 'nhpsa'
$applocation = "app/$($args[0] ?? 'dev')"

Write-Host "uploading to: $applocation"

# upload the app code to blob storage
az storage blob upload-batch --account-name $saname --source model --destination $applocation --destination-path model --auth-mode login
az storage blob upload --account-name $saname -c $applocation -f run_model.py --auth-mode login