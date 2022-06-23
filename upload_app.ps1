$saname = 'nhpsa'

# upload the app code to blob storage
az storage blob upload-batch --account-name $saname --source model --destination app --destination-path model --auth-mode login
az storage blob upload --account-name $saname -c app -f run_model.py --auth-mode login