$saname = 'nhpsa'
$app_location = "app/$($args[0] ?? 'dev')"

Write-Host "uploading to: $app_location"

# upload the app code to blob storage
az storage blob upload-batch --account-name $saname --source model --destination $app_location --destination-path model --auth-mode login --overwrite
az storage blob upload --account-name $saname -c $app_location -f run_model.py --auth-mode login --overwrite